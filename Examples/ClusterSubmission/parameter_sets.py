# This is set of parameters that was selected based on a small scale sensitivity analysis using Morris screening.

from besos.parameters import (
    wwr,
    RangeParameter,
    FieldSelector,
    FilterSelector,
    GenericSelector,
    Parameter,
    expand_plist,
)


# 4 Parameters
def parameter_set(number_of_pars=4):
    if number_of_pars == 4:
        parameters = expand_plist(
            {
                "NonRes Fixed Assembly Window": {
                    "Solar Heat Gain Coefficient": (0.01, 0.99)
                }
            }
        )

        parameters.append(
            Parameter(
                selector=FieldSelector(
                    class_name="ElectricEquipment",
                    object_name="*",
                    field_name="Watts per Zone Floor Area",
                ),
                value_descriptor=RangeParameter(min_val=10, max_val=15),
            )
        )

        parameters.append(
            Parameter(
                selector=FieldSelector(
                    class_name="Lights",
                    object_name="*",
                    field_name="Watts per Zone Floor Area",
                ),
                value_descriptor=RangeParameter(min_val=10, max_val=15),
            )
        )

        parameters.append(wwr())

    elif number_of_pars == 8:

        parameters = expand_plist(
            {
                # Wall Materials
                "Mass NonRes Wall Insulation": {"Conductivity": (0.02, 0.2)},
                "AtticFloor NonRes Insulation": {"Thickness": (0.1, 0.3)},
                # Window Materials
                "NonRes Fixed Assembly Window": {
                    "U-Factor": (0.1, 5),
                    "Solar Heat Gain Coefficient": (0.01, 0.99),
                },
            }
        )

        parameters[0].name = "Wall conductivity"
        parameters[1].name = "Attic thickness"

        # parameters.append(Parameter(
        #    selector = FieldSelector(class_name  = 'ZoneInfiltration:DesignFlowRate',
        #                             object_name = '*',
        #                             field_name  = 'Flow per Exterior Surface Area' ),
        #                            value_descriptor = RangeParameter(min_val=0, max_val=0.002),
        #                            name='Infiltration/Area'))

        parameters.append(
            Parameter(
                selector=FieldSelector(
                    class_name="ElectricEquipment",
                    object_name="*",
                    field_name="Watts per Zone Floor Area",
                ),
                value_descriptor=RangeParameter(min_val=10, max_val=15),
            )
        )

        parameters.append(
            Parameter(
                selector=FieldSelector(
                    class_name="Lights",
                    object_name="*",
                    field_name="Watts per Zone Floor Area",
                ),
                value_descriptor=RangeParameter(min_val=10, max_val=15),
            )
        )

        parameters.append(wwr())
    elif number_of_pars == 12:
        parameters = expand_plist(
            {
                # Wall Materials
                "Mass NonRes Wall Insulation": {
                    "Thickness": (0.01, 0.5),
                    "Conductivity": (0.02, 0.2),
                },
                "AtticFloor NonRes Insulation": {"Thickness": (0.1, 0.3)},
                # Window Materials
                "NonRes Fixed Assembly Window": {
                    "U-Factor": (0.1, 5),
                    "Solar Heat Gain Coefficient": (0.01, 0.99),
                },
                # Orientation
                "Ref Bldg Small Office New2004_v1.3_5.0": {"North Axis": (0, 360)},
                "SWHSys1 Water Heater": {"Heater Thermal Efficiency": (0.7, 0.95)},
            }
        )

        parameters[0].name = "Thickness"
        parameters[2].name = "Attic thickness"

        # parameters.append(Parameter(selector = FieldSelector(class_name  = 'ZoneInfiltration:DesignFlowRate',
        #                                                     object_name = '*',
        #                                                     field_name  = 'Flow per Exterior Surface Area' ),
        #                            value_descriptor = RangeParameter(min_val=0, max_val=0.002),
        #                            name='Infiltration/Area'))

        parameters.append(
            Parameter(
                selector=FieldSelector(
                    class_name="ElectricEquipment",
                    object_name="*",
                    field_name="Watts per Zone Floor Area",
                ),
                value_descriptor=RangeParameter(min_val=10, max_val=15),
            )
        )

        parameters.append(
            Parameter(
                selector=FieldSelector(
                    class_name="Lights",
                    object_name="*",
                    field_name="Watts per Zone Floor Area",
                ),
                value_descriptor=RangeParameter(min_val=10, max_val=15),
            )
        )

        parameters.append(
            Parameter(
                selector=FieldSelector(
                    class_name="Coil:Heating:Fuel",
                    object_name="*",
                    field_name="Burner Efficiency",
                ),
                value_descriptor=RangeParameter(min_val=0.7, max_val=0.9),
            )
        )

        parameters.append(wwr())

    return parameters
