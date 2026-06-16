# Zone-Raw Diagnostic Run QA

- Run directory: `Probabilities_ENB/paperA_rebuild/runs/diagnostic_reference_zone_raw_full`
- Trace CSVs: 144 / 144
- EnergyPlus `.err` files: 144 / 144
- EnergyPlus `.end` files: 144 / 144
- Inventory complete: True
- Schema OK: True
- Annual row counts OK: True
- Occupied raw zone values OK: True
- EnergyPlus completed without fatal errors: True
- Cases with Severe reports: 16

## Severe Cases

```csv
case,severe_lines,warmup_convergence_lines,summary_severe,summary
ahmedabad_ssp245_baseline_2020s_typical_2025,1,1,1,************* EnergyPlus Completed Successfully-- 28228 Warning; 1 Severe Errors; Elapsed Time=00hr 00min 58.60sec
ahmedabad_ssp585_near_2030s_heatwave_extreme_2037,1,1,1,************* EnergyPlus Completed Successfully-- 28293 Warning; 1 Severe Errors; Elapsed Time=00hr 00min 56.14sec
ahmedabad_ssp585_near_2030s_hot_2037,1,1,1,************* EnergyPlus Completed Successfully-- 28293 Warning; 1 Severe Errors; Elapsed Time=00hr 00min 56.11sec
ahmedabad_ssp585_near_2030s_typical_2031,1,1,1,************* EnergyPlus Completed Successfully-- 28247 Warning; 1 Severe Errors; Elapsed Time=00hr 00min 56.11sec
beijing_ssp585_late_2080s_typical_2084,2,2,2,************* EnergyPlus Completed Successfully-- 28230 Warning; 2 Severe Errors; Elapsed Time=00hr 00min 57.87sec
guangzhou_ssp245_late_2080s_heatwave_extreme_2088,1,1,1,************* EnergyPlus Completed Successfully-- 28286 Warning; 1 Severe Errors; Elapsed Time=00hr 00min 56.63sec
guangzhou_ssp245_mid_2050s_heatwave_extreme_2055,2,2,2,************* EnergyPlus Completed Successfully-- 28339 Warning; 2 Severe Errors; Elapsed Time=00hr 00min 56.89sec
guangzhou_ssp245_mid_2050s_hot_2051,2,2,2,************* EnergyPlus Completed Successfully-- 28247 Warning; 2 Severe Errors; Elapsed Time=00hr 00min 58.05sec
guangzhou_ssp245_mid_2050s_typical_2054,2,2,2,************* EnergyPlus Completed Successfully-- 28364 Warning; 2 Severe Errors; Elapsed Time=00hr 00min 58.09sec
houston_ssp245_mid_2050s_hot_2056,2,2,2,************* EnergyPlus Completed Successfully-- 28225 Warning; 2 Severe Errors; Elapsed Time=00hr 00min 56.53sec
houston_ssp585_late_2080s_heatwave_extreme_2086,2,2,2,************* EnergyPlus Completed Successfully-- 28399 Warning; 2 Severe Errors; Elapsed Time=00hr 00min 57.28sec
kolkata_ssp245_late_2080s_heatwave_extreme_2086,1,1,1,************* EnergyPlus Completed Successfully-- 28251 Warning; 1 Severe Errors; Elapsed Time=00hr 00min 57.16sec
kolkata_ssp245_mid_2050s_heatwave_extreme_2053,1,1,1,************* EnergyPlus Completed Successfully-- 28325 Warning; 1 Severe Errors; Elapsed Time=00hr 00min 58.70sec
kolkata_ssp245_mid_2050s_hot_2053,1,1,1,************* EnergyPlus Completed Successfully-- 28325 Warning; 1 Severe Errors; Elapsed Time=00hr 00min 57.82sec
kolkata_ssp585_baseline_2020s_typical_2029,2,2,2,************* EnergyPlus Completed Successfully-- 28302 Warning; 2 Severe Errors; Elapsed Time=00hr 00min 58.07sec
phoenix_ssp585_mid_2050s_typical_2050,2,2,2,************* EnergyPlus Completed Successfully-- 28354 Warning; 2 Severe Errors; Elapsed Time=00hr 01min  2.37sec
```

## Trace Audit Head

```csv
trace,rows,occupied_rows,columns,zone_raw_fields,header_matches_reference,expected_rows,expected_occupied_rows,missing_raw_occupied,nonfinite_raw_occupied,constant_raw_columns,zero_byte
ahmedabad_ssp245_baseline_2020s_heatwave_extreme_2029_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp245_baseline_2020s_hot_2029_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp245_baseline_2020s_typical_2025_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp245_late_2080s_heatwave_extreme_2081_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp245_late_2080s_hot_2081_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp245_late_2080s_typical_2086_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp245_mid_2050s_heatwave_extreme_2059_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp245_mid_2050s_hot_2059_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp245_mid_2050s_typical_2058_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp245_near_2030s_heatwave_extreme_2034_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp245_near_2030s_hot_2034_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp245_near_2030s_typical_2032_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp585_baseline_2020s_heatwave_extreme_2027_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp585_baseline_2020s_hot_2027_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp585_baseline_2020s_typical_2025_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp585_late_2080s_heatwave_extreme_2088_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp585_late_2080s_hot_2088_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp585_late_2080s_typical_2087_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp585_mid_2050s_heatwave_extreme_2051_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
ahmedabad_ssp585_mid_2050s_hot_2050_diagnostic_reference.csv,35040,16704,162,45,True,True,True,0,0,0,False
```
