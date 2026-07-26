{
  "configuration_file": "configs/case_leo400_gmat_matched.json",
  "configuration_sha256": "181d71fb24bad244d6c7ab8fe5fd9bfe1fa13b0431b60b6ac29c16c82bcf4037",
  "earth_model": {
    "constants_reference": "GMAT R2026a bundled application/data/gravity/earth/JGM2.cof: GM=3.986004415e14 m^3/s^2, reference radius=6378136.3 m, J2 derived from fully-normalized C20 as -sqrt(5)*C20.",
    "earth_rotation_rate_rad_s": 7.292115e-05,
    "equatorial_radius_km": 6378.1363,
    "gravitational_parameter_km3_s2": 398600.4415,
    "j2": 0.001082626724392697,
    "name": "GMAT JGM2 matched constants"
  },
  "epoch_utc": "2026-01-01T00:00:00Z",
  "external_validation": {
    "acceleration_diagnostic": {
      "duration_seconds": 86400.0,
      "enabled": true,
      "isolation_method": "degree2_order0_minus_degree0_order0_at_shared_state",
      "sample_count": 25,
      "state_source": "single_degree2_propagated_spacecraft",
      "threshold_status": "provisional_until_first_real_acceleration_report",
      "thresholds": {
        "maximum_relative_difference": 1e-05,
        "maximum_vector_difference_km_s2": 1e-10,
        "point_mass_maximum_vector_difference_km_s2": 1e-10
      }
    },
    "duration_seconds": 86400.0,
    "enabled": true,
    "ephemeris_format": "STK-TimePosVel",
    "frame": "EarthMJ2000Eq",
    "gmat_accuracy": 1e-13,
    "gmat_initial_step_seconds": 60.0,
    "gmat_integrator": "PrinceDormand78",
    "gmat_maximum_step_seconds": 60.0,
    "gravity_file": "JGM2.cof",
    "j2_frame_models": {
      "fixed_axis": {
        "model_name": "numerical_j2_fixed_axis",
        "purpose": "textbook_analysis_and_analytical_raan_comparison",
        "symmetry_axis": "EarthMJ2000Eq z-axis"
      },
      "gmat_matched": {
        "implementation": "ERFA pnm80 evaluated in TT",
        "model_name": "numerical_j2_gmat_matched",
        "purpose": "GMAT_JGM2_degree2_order0_external_validation",
        "remaining_difference_sources": [
          "Exact GMAT EOP file and polar-motion treatment are not reproduced.",
          "EarthMJ2000Eq and ERFA celestial-axis realization differences remain measurable.",
          "Independent integrators and GMAT ephemeris interpolation remain active."
        ],
        "symmetry_axis": "IAU-1976/FK5 precession plus IAU-1980 nutation true pole"
      }
    },
    "output_step_seconds": 60.0,
    "python_models": [
      "numerical_two_body",
      "numerical_j2_fixed_axis",
      "numerical_j2_gmat_matched"
    ],
    "scientific_cautions": [
      "The fixed-axis J2 result is diagnostic-only for GMAT comparison and remains valid for the stated textbook assumption.",
      "Thresholds must not be loosened merely to obtain a pass.",
      "Earth constants, force-model degree/order, frame, epoch, initial Cartesian state, and output epochs must match.",
      "Acceleration isolation requires degree 0 and degree 2 evaluation at the exact same spacecraft state.",
      "STK ephemeris output uses interpolation at fixed 60-second epochs.",
      "External validation compares independently developed implementations, not measured orbit truth."
    ],
    "short_arc": {
      "duration_seconds": 600.0,
      "enabled": true,
      "output_step_seconds": 1.0,
      "threshold_status": "provisional_until_first_real_short_arc",
      "thresholds": {
        "gmat_matched_maximum_position_difference_m": 1.0,
        "gmat_matched_maximum_velocity_difference_mm_s": 1.0
      }
    },
    "threshold_status": "provisional_diagnostic_after_first_gmat_run",
    "thresholds": {
      "initial_position_difference_m": 0.001,
      "initial_velocity_difference_mm_s": 0.001,
      "j2_maximum_position_difference_m": 500.0,
      "j2_maximum_velocity_difference_mm_s": 500.0,
      "two_body_maximum_position_difference_m": 50.0,
      "two_body_maximum_velocity_difference_mm_s": 50.0
    },
    "tool": "GMAT",
    "tool_release_status": "NASA_GSFC_release_2026-05-06",
    "tool_version": "R2026a"
  },
  "frame": "EarthMJ2000Eq",
  "generated_files": {
    "acceleration_diagnostic_report": "data/reference/gmat/output/CASE_LEO400_GMAT_ACCELERATION_DIAGNOSTIC.csv",
    "acceleration_diagnostic_script": "data/reference/gmat/scripts/CASE_LEO400_GMAT_ACCELERATION_DIAGNOSTIC.script",
    "j2_ephemeris": "data/reference/gmat/output/CASE_LEO400_GMAT_J2.e",
    "j2_script": "data/reference/gmat/scripts/CASE_LEO400_GMAT_J2.script",
    "j2_short_arc_ephemeris": "data/reference/gmat/output/CASE_LEO400_GMAT_J2_SHORT_ARC.e",
    "j2_short_arc_script": "data/reference/gmat/scripts/CASE_LEO400_GMAT_J2_SHORT_ARC.script",
    "two_body_ephemeris": "data/reference/gmat/output/CASE_LEO400_GMAT_TWO_BODY.e",
    "two_body_script": "data/reference/gmat/scripts/CASE_LEO400_GMAT_TWO_BODY.script"
  },
  "initial_position_km": [
    4791.244801771759,
    3981.8438470941833,
    2653.3345450508014
  ],
  "initial_velocity_km_s": [
    -5.018943360359614,
    2.567534615772381,
    5.209846002826843
  ],
  "research_core_version": "1A.8.2",
  "script_sha256": {
    "acceleration_diagnostic": "501ec70ea41ddbc3dcfc4e50a96c60932d14826d7222b3f2f5a263592fe0d5ae",
    "j2": "01439a7ee0bcf86c3a919e6108c89c0449dcd70edb06de55c3df93072b9490fb",
    "j2_short_arc": "47f10b781afa6fbfb3435c601648fe5a7c496df7f6d3524bb66c2c448ac028b2",
    "two_body": "341f7e87903cf39b0829173a50989bd1a2731ad232f317498998ca2a51fd808c"
  },
  "status": "scripts_prepared_gmat_execution_pending",
  "target_tool": "GMAT",
  "target_tool_version": "R2026a"
}
