import os

import numpy as np
import openmdao.api as om
from scipy.interpolate import PchipInterpolator


class ParametrizeBladeAero(om.ExplicitComponent):
    # Openmdao component to parameterize distributed quantities for the outer shape of the wind turbine rotor blades
    def initialize(self):
        self.options.declare("rotorse_options")
        self.options.declare("opt_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.opt_options = self.options["opt_options"]
        n_span = rotorse_options["n_span"]
        self.n_opt_twist = n_opt_twist = self.opt_options["design_variables"]["blade"]["aero_shape"]["twist"]["n_opt"]
        self.n_opt_chord = n_opt_chord = self.opt_options["design_variables"]["blade"]["aero_shape"]["chord"]["n_opt"]
        self.n_opt_length_te = n_opt_length_te = self.opt_options["design_variables"]["blade"]["aero_shape"]["length_te"]["n_opt"]

        # if (self.opt_options["design_variables"]["blade"]["aero_shape"]["chord"]["flag"]
        #         and self.opt_options["design_variables"]["blade"]["aero_shape"]["length_te"]["flag"]):
        #     raise NotImplementedError("you've specified to optimize both chord and trailing edge length at the same time.")

        # Inputs
        self.add_input(
            "s",
            val=np.zeros(n_span),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)",
        )
        # # Blade twist
        # self.add_input(
        #     "twist_original",
        #     val=np.zeros(n_span),
        #     units="rad",
        #     desc="1D array of the twist values defined along blade span. The twist is the one defined in the yaml.",
        # )
        self.add_input(
            "s_opt_twist",
            val=np.zeros(n_opt_twist),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade twist angle",
        )
        self.add_input(
            "twist_opt",
            val=np.ones(n_opt_twist),
            units="rad",
            desc="1D array of the twist angle being optimized at the n_opt locations.",
        )
        # Blade chord
        # self.add_input(
        #     "chord_original",
        #     val=np.zeros(n_span),
        #     units="m",
        #     desc="1D array of the chord values defined along blade span. The chord is the one defined in the yaml.",
        # )
        self.add_input(
            "s_opt_chord",
            val=np.zeros(n_opt_chord),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade chord",
        )
        self.add_input(
            "chord_opt",
            val=np.ones(n_opt_chord),
            units="m",
            desc="1D array of the chord being optimized at the n_opt locations.",
        )

        self.add_input(
            "pitch_axis",
            val=np.zeros(n_span),
            desc="1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.",
        )

        self.add_input(
            "s_opt_length_te",
            val=np.zeros(n_opt_chord),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade trailing edge length"
        )
        self.add_input(
            "length_te_opt",
            val=np.ones(n_opt_length_te),
            units="m",
            desc="1D array of the chord being optimized at the n_opt locations.",
        )

        # Outputs
        self.add_output(
            "twist_param",
            val=np.zeros(n_span),
            units="rad",
            desc="1D array of the twist values defined along blade span. The twist is the result of the parameterization.",
        )
        self.add_output(
            "chord_param",
            val=np.zeros(n_span),
            units="m",
            desc="1D array of the chord values defined along blade span. The chord is the result of the parameterization.",
        )
        self.add_output(
            "pitch_axis_param",
            val=np.zeros(n_span),
            desc="1D array of the pitch axis locations defined along blade span. The pitch axis is the result of the parameterization.",
        )
        self.add_output(
            "max_chord_constr",
            val=np.zeros(n_opt_chord),
            desc="1D array of the ratio between chord values and maximum chord along blade span.",
        )

    def compute(self, inputs, outputs):

        spline = PchipInterpolator
        twist_spline = spline(inputs["s_opt_twist"], inputs["twist_opt"])
        outputs["twist_param"] = twist_spline(inputs["s"])
        chord_spline = spline(inputs["s_opt_chord"], inputs["chord_opt"])
        if self.opt_options["design_variables"]["blade"]["aero_shape"]["length_te"]["flag"]:
            chord_init= chord_spline(inputs["s"])
            pa_init= inputs["pitch_axis"]
            length_le_init= pa_init*chord_init # leading edge length
            length_te_init= inputs["length_te_opt"] # trailing edge lencth

            outputs["chord_param"] = length_le_init + length_te_init
            outputs["pitch_axis_param"] = length_le_init/outputs["chord_param"]
        else:
            outputs["chord_param"] = chord_spline(inputs["s"])
            outputs["pitch_axis_param"] = inputs["pitch_axis"]

        chord_opt = spline(inputs["s"], outputs["chord_param"])
        max_chord = self.opt_options["constraints"]["blade"]["chord"]["max"]
        outputs["max_chord_constr"] = chord_opt(inputs["s_opt_chord"]) / max_chord


class ParametrizeBladeStruct(om.ExplicitComponent):
    # Openmdao component to parameterize distributed quantities for the structural design of the wind turbine rotor blades
    def initialize(self):
        self.options.declare("rotorse_options")
        self.options.declare("opt_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.opt_options = opt_options = self.options["opt_options"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_layers = n_layers = rotorse_options["n_layers"]
        self.n_opt_spar_cap_ss = n_opt_spar_cap_ss = opt_options["design_variables"]["blade"]["structure"][
            "spar_cap_ss"
        ]["n_opt"]
        self.n_opt_spar_cap_ps = n_opt_spar_cap_ps = opt_options["design_variables"]["blade"]["structure"][
            "spar_cap_ps"
        ]["n_opt"]
        self.n_opt_te_ss = n_opt_te_ss = opt_options["design_variables"]["blade"]["structure"]["te_ss"]["n_opt"]
        self.n_opt_te_ps = n_opt_te_ps = opt_options["design_variables"]["blade"]["structure"]["te_ps"]["n_opt"]

        # Inputs
        self.add_input(
            "s",
            val=np.zeros(n_span),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)",
        )
        self.add_input(
            "layer_thickness_original",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the thickness of the layers of the blade structure. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        # Blade spar suction side
        self.add_input(
            "s_opt_spar_cap_ss",
            val=np.zeros(n_opt_spar_cap_ss),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade spar cap suction side",
        )
        self.add_input(
            "spar_cap_ss_opt",
            val=np.ones(n_opt_spar_cap_ss),
            units="m",
            desc="1D array of the the blade spanwise distribution of the spar caps suction side being optimized",
        )
        # Blade spar suction side
        self.add_input(
            "s_opt_spar_cap_ps",
            val=np.zeros(n_opt_spar_cap_ps),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade spar cap pressure side",
        )
        self.add_input(
            "spar_cap_ps_opt",
            val=np.ones(n_opt_spar_cap_ps),
            units="m",
            desc="1D array of the the blade spanwise distribution of the spar caps pressure side being optimized",
        )
        # Blade TE suction side
        self.add_input(
            "s_opt_te_ss",
            val=np.zeros(n_opt_te_ss),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade trailing edge suction side",
        )
        self.add_input(
            "te_ss_opt",
            val=np.ones(n_opt_te_ss),
            units="m",
            desc="1D array of the the blade spanwise distribution of the trailing edges suction side being optimized",
        )
        # Blade TE suction side
        self.add_input(
            "s_opt_te_ps",
            val=np.zeros(n_opt_te_ps),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade trailing edge pressure side",
        )
        self.add_input(
            "te_ps_opt",
            val=np.ones(n_opt_te_ps),
            units="m",
            desc="1D array of the the blade spanwise distribution of the trailing edges pressure side being optimized",
        )

        # Outputs
        self.add_output(
            "layer_thickness_param",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the thickness of the layers of the blade structure after the parametrization. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )

    def compute(self, inputs, outputs):

        layer_name = self.options["rotorse_options"]["layer_name"]

        spar_cap_ss_name = self.options["rotorse_options"]["spar_cap_ss"]
        spar_cap_ps_name = self.options["rotorse_options"]["spar_cap_ps"]

        ss_before_ps = False
        opt_ss = self.opt_options["design_variables"]["blade"]["structure"]["spar_cap_ss"]["flag"]
        opt_ps = self.opt_options["design_variables"]["blade"]["structure"]["spar_cap_ss"]["flag"]
        for i in range(self.n_layers):
            if layer_name[i] == spar_cap_ss_name and opt_ss and opt_ps:
                opt_m_interp = np.interp(inputs["s"], inputs["s_opt_spar_cap_ss"], inputs["spar_cap_ss_opt"])
                ss_before_ps = True
            elif layer_name[i] == spar_cap_ps_name and opt_ss and opt_ps:
                if (
                    self.opt_options["design_variables"]["blade"]["structure"]["spar_cap_ps"]["equal_to_suction"]
                    == False
                ) or ss_before_ps == False:
                    opt_m_interp = np.interp(inputs["s"], inputs["s_opt_spar_cap_ps"], inputs["spar_cap_ps_opt"])
                else:
                    opt_m_interp = np.interp(inputs["s"], inputs["s_opt_spar_cap_ss"], inputs["spar_cap_ss_opt"])
            else:
                opt_m_interp = inputs["layer_thickness_original"][i, :]

            outputs["layer_thickness_param"][i, :] = opt_m_interp

        te_ss_name = self.options["rotorse_options"]["te_ss"]
        te_ps_name = self.options["rotorse_options"]["te_ps"]

        ss_before_ps = False
        opt_ss = self.opt_options["design_variables"]["blade"]["structure"]["te_ss"]["flag"]
        opt_ps = self.opt_options["design_variables"]["blade"]["structure"]["te_ss"]["flag"]
        for i in range(self.n_layers):
            if layer_name[i] == te_ss_name and opt_ss and opt_ps:
                opt_m_interp = np.interp(inputs["s"], inputs["s_opt_te_ss"], inputs["te_ss_opt"])
                ss_before_ps = True
            elif layer_name[i] == te_ps_name and opt_ss and opt_ps:
                if (
                    self.opt_options["design_variables"]["blade"]["structure"]["te_ps"]["equal_to_suction"] == False
                ) or ss_before_ps == False:
                    opt_m_interp = np.interp(inputs["s"], inputs["s_opt_te_ps"], inputs["te_ps_opt"])
                else:
                    opt_m_interp = np.interp(inputs["s"], inputs["s_opt_te_ss"], inputs["te_ss_opt"])
            else:
                opt_m_interp = outputs["layer_thickness_param"][i, :]

            outputs["layer_thickness_param"][i, :] = opt_m_interp


class ComputeReynolds(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("n_span")

    def setup(self):
        n_span = self.options["n_span"]

        self.add_input("rho", val=0.0, units="kg/m**3")
        self.add_input("mu", val=1.81e-5, units="kg/(m*s)", desc="Dynamic viscosity of air")
        self.add_input("local_airfoil_velocities", val=np.zeros((n_span)), units="m/s")
        self.add_input("chord", val=np.zeros((n_span)), units="m")
        self.add_output("Re", val=np.zeros((n_span)), ref=1.0e6)

    def compute(self, inputs, outputs):
        outputs["Re"] = np.nan_to_num(
            inputs["rho"] * inputs["local_airfoil_velocities"] * inputs["chord"] / inputs["mu"]
        )
