---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:11100
- loss:MultipleNegativesRankingLoss
base_model: nreimers/MiniLM-L6-H384-uncased
widget:
- source_sentence: 'I''m looking at the digital‑twin interface for monitoring nutrient
    concentrations affecting aquatic ecosystems and need details on the phosphorus
    monitoring unit: it uses a phosense‑3000 sensor model that reports a phosphate
    concentration of 0.12, has a specified accuracy of 0.02, and the maintenance logs
    show the last service was performed on 2024‑09‑15T00:00:00Z, matching the planned
    interval, so can you confirm that this hardware and its documented accuracy enable
    reliable low‑level phosphate detection and that the maintenance schedule is being
    met to maintain measurement fidelity?'
  sentences:
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Models a sensor that measures dissolved
    phosphate in the river, providing key data for nutrient loading studies. It records
    device metadata such as model and last maintenance date, and flags any measurement
    anomalies. The twin supports integration with watershed management platforms.'',
    ''contents'': [{''@type'': ''Property'', ''name'': ''phosphateConcentration'',
    ''schema'': ''double'', ''description'': ''Current phosphate concentration in
    mg/L''}, {''@type'': ''Property'', ''name'': ''sensorModel'', ''schema'': ''string'',
    ''description'': ''Manufacturer model identifier''}, {''@type'': ''Property'',
    ''name'': ''lastMaintenance'', ''schema'': ''string'', ''description'': ''ISO‑8601
    date of the most recent maintenance''}, {''@type'': ''Property'', ''name'': ''accuracy'',
    ''schema'': ''double'', ''description'': ''Stated measurement accuracy as a percentage''},
    {''@type'': ''Telemetry'', ''name'': ''phosphateLevel'', ''schema'': ''double'',
    ''description'': ''Real‑time phosphate level in mg/L''}, {''@type'': ''Telemetry'',
    ''name'': ''waterTemperature'', ''schema'': ''double'', ''description'': ''Ambient
    water temperature in °C''}, {''@type'': ''Telemetry'', ''name'': ''signalStrength'',
    ''schema'': ''double'', ''description'': ''Signal strength of the wireless transmission
    in dBm''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Represents a sensor that continuously
    measures nitrate concentrations in river water. The twin captures calibration
    metadata and operational status to support quality assurance. It enables downstream
    analytics for eutrophication risk assessment.'', ''contents'': [{''@type'': ''Property'',
    ''name'': ''nitrateConcentration'', ''schema'': ''double'', ''description'': ''Current
    nitrate concentration in mg/L''}, {''@type'': ''Property'', ''name'': ''sensorCalibrationDate'',
    ''schema'': ''string'', ''description'': ''ISO‑8601 date of the last calibration''},
    {''@type'': ''Property'', ''name'': ''measurementDepth'', ''schema'': ''double'',
    ''description'': ''Depth at which the sensor samples, in meters''}, {''@type'':
    ''Property'', ''name'': ''status'', ''schema'': ''string'', ''description'': ''Operational
    status (OK, Warning, Error)''}, {''@type'': ''Telemetry'', ''name'': ''nitrateLevel'',
    ''schema'': ''double'', ''description'': ''Real‑time nitrate level reading in
    mg/L''}, {''@type'': ''Telemetry'', ''name'': ''waterTemperature'', ''schema'':
    ''double'', ''description'': ''Water temperature at the sensor location in °C''},
    {''@type'': ''Telemetry'', ''name'': ''batteryVoltage'', ''schema'': ''double'',
    ''description'': ''Battery voltage supplying the sensor''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Computes the drag coefficient using
    Reynolds and Mach numbers together with turbulence intensity. Implements empirical
    and CFD‑based formulas to adapt to varying flow regimes. The calculator updates
    the AeroDragModel in near real time.'', ''contents'': [{''@type'': ''Property'',
    ''name'': ''reynoldsNumber'', ''schema'': ''double''}, {''@type'': ''Property'',
    ''name'': ''machNumber'', ''schema'': ''double''}, {''@type'': ''Property'', ''name'':
    ''turbulenceIntensity'', ''schema'': ''double''}, {''@type'': ''Telemetry'', ''name'':
    ''calculatedDragCoeff'', ''schema'': ''double''}, {''@type'': ''Telemetry'', ''name'':
    ''sensitivityAnalysis'', ''schema'': ''double''}]}'
- source_sentence: Can you explain the digital‑twin interface that tracks nitrate‑concentration
    spread from agricultural sources, which models a chemical processing unit with
    a reaction‑rate constant of 0.0043, a temperature‑sensitivity factor of 0.0185
    per degree, and operates at a pH of 7.2, so I can understand how these three parameters
    enable precise simulation of reaction dynamics when temperature varies and how
    engineers can use them to predict acceleration or deceleration of the process,
    assess material compatibility and corrosion risk, and develop control strategies
    for performance optimization?
  sentences:
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Simulates advective‑dispersive transport
    of nitrate through the aquifer matrix. It uses hydraulic properties and spatial
    discretization to predict plume migration over time. The model exchanges concentration
    fields with upstream source and downstream reaction components.'', ''contents'':
    [{''@type'': ''Property'', ''name'': ''hydraulicConductivity'', ''schema'': ''double'',
    ''unit'': ''m/day'', ''description'': ''Average hydraulic conductivity of the
    aquifer''}, {''@type'': ''Property'', ''name'': ''porosity'', ''schema'': ''double'',
    ''description'': ''Effective porosity fraction''}, {''@type'': ''Property'', ''name'':
    ''gradient'', ''schema'': ''double'', ''unit'': ''m/m'', ''description'': ''Hydraulic
    gradient driving flow''}, {''@type'': ''Telemetry'', ''name'': ''plumeFrontPosition'',
    ''schema'': ''double'', ''unit'': ''m'', ''description'': ''Distance of the leading
    edge of the nitrate plume from the source''}, {''@type'': ''Telemetry'', ''name'':
    ''nitrateConcentration'', ''schema'': ''double'', ''unit'': ''ppm'', ''description'':
    ''Average nitrate concentration at a monitoring point''}, {''@type'': ''Telemetry'',
    ''name'': ''travelTime'', ''schema'': ''double'', ''unit'': ''days'', ''description'':
    ''Estimated travel time for water parcels from source to observation location''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Applies biogeochemical reactions that
    transform nitrate within the plume. It accounts for denitrification, nitrification,
    and pH‑dependent kinetics. The model updates concentration fields based on temperature
    and substrate availability.'', ''contents'': [{''@type'': ''Property'', ''name'':
    ''reactionRateConstant'', ''schema'': ''double'', ''unit'': ''1/day'', ''description'':
    ''Base first‑order denitrification rate constant''}, {''@type'': ''Property'',
    ''name'': ''temperatureSensitivity'', ''schema'': ''double'', ''unit'': ''1/C'',
    ''description'': ''Arrhenius temperature coefficient''}, {''@type'': ''Property'',
    ''name'': ''pH'', ''schema'': ''double'', ''description'': ''Current groundwater
    pH influencing reaction pathways''}, {''@type'': ''Telemetry'', ''name'': ''denitrificationRate'',
    ''schema'': ''double'', ''unit'': ''kg/ha/day'', ''description'': ''Mass of nitrate
    removed per day by denitrification''}, {''@type'': ''Telemetry'', ''name'': ''ammoniaConcentration'',
    ''schema'': ''double'', ''unit'': ''ppm'', ''description'': ''Concentration of
    ammonia produced during nitrification''}, {''@type'': ''Telemetry'', ''name'':
    ''nitrateReduction'', ''schema'': ''double'', ''unit'': ''%'', ''description'':
    ''Percent reduction of nitrate relative to input load''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Implements the rainflow counting algorithm
    and material-specific S‑N curves to predict remaining cycles to failure. The model
    updates damage accumulation whenever new stress data arrive.'', ''contents'':
    [{''@type'': ''Property'', ''name'': ''modelVersion'', ''schema'': ''string'',
    ''description'': ''Semantic version of the fatigue model code.''}, {''@type'':
    ''Property'', ''name'': ''materialFatigueLimit'', ''schema'': ''double'', ''description'':
    ''Endurance limit of the cable material in MPa.''}, {''@type'': ''Property'',
    ''name'': ''damageExponent'', ''schema'': ''double'', ''description'': ''Exponent
    used in the damage accumulation law.''}, {''@type'': ''Property'', ''name'': ''useRainflowCounting'',
    ''schema'': ''boolean'', ''description'': ''Flag to enable rainflow counting for
    cycle identification.''}, {''@type'': ''Telemetry'', ''name'': ''cumulativeDamage'',
    ''schema'': ''double'', ''description'': ''Total damage accumulated (0‑1 scale).''},
    {''@type'': ''Telemetry'', ''name'': ''predictedCyclesToFailure'', ''schema'':
    ''double'', ''description'': ''Estimated number of load cycles remaining before
    failure.''}, {''@type'': ''Telemetry'', ''name'': ''modelStatus'', ''schema'':
    ''string'', ''description'': ''Current operational status of the model (e.g.,
    Ready, Updating).''}]}'
- source_sentence: The interface for twin injection mold cooling channels, aimed at
    optimizing temperature uniformity, is configured to maintain a setpoint temperature
    of 185.5 °C using PID control, with a proportional gain (Kp) of 2.8 for a strong
    immediate response, an integral gain (Ki) of 0.12 to eliminate steady‑state error,
    and a derivative gain (Kd) of 0.015 to damp overshoot, thereby providing precise,
    stable temperature regulation with minimal oscillation and settling time for applications
    requiring tight thermal tolerance and reliable performance.
  sentences:
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Defines the policy rules that govern
    how and when robots may charge. Parameters include minimum and maximum battery
    thresholds, allowed charging hours, and an emergency override flag for critical
    situations. The twin provides a central reference for the scheduler to enforce
    consistent charging behavior across the fleet.'', ''contents'': [{''@type'': ''Property'',
    ''name'': ''policyId'', ''schema'': ''string'', ''writable'': True}, {''@type'':
    ''Property'', ''name'': ''minBatteryThresholdPercent'', ''schema'': ''integer''},
    {''@type'': ''Property'', ''name'': ''maxBatteryThresholdPercent'', ''schema'':
    ''integer''}, {''@type'': ''Property'', ''name'': ''allowedChargingHours'', ''schema'':
    ''string''}, {''@type'': ''Property'', ''name'': ''emergencyOverrideEnabled'',
    ''schema'': ''boolean''}, {''@type'': ''Telemetry'', ''name'': ''policyUpdated'',
    ''schema'': ''string''}, {''@type'': ''Telemetry'', ''name'': ''overrideActivated'',
    ''schema'': ''string''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Models the physical geometry of each
    cooling channel within the injection mold. Captures dimensions that affect heat
    extraction and flow distribution, enabling simulation of thermal performance.'',
    ''contents'': [{''@type'': ''Property'', ''name'': ''channelDiameter'', ''schema'':
    ''double'', ''description'': ''Diameter of a single cooling channel in millimeters.''},
    {''@type'': ''Property'', ''name'': ''channelLength'', ''schema'': ''double'',
    ''description'': ''Length of the cooling channel measured along the flow path
    in millimeters.''}, {''@type'': ''Property'', ''name'': ''numberOfChannels'',
    ''schema'': ''integer'', ''description'': ''Total count of cooling channels in
    the mold.''}, {''@type'': ''Property'', ''name'': ''channelPitch'', ''schema'':
    ''double'', ''description'': ''Center-to-center spacing between adjacent channels
    in millimeters.''}, {''@type'': ''Telemetry'', ''name'': ''pressureDrop'', ''schema'':
    ''double'', ''description'': ''Measured pressure drop across the channel network
    in bar.''}, {''@type'': ''Telemetry'', ''name'': ''flowUniformityIndex'', ''schema'':
    ''double'', ''description'': ''Index (0‑1) indicating uniformity of coolant flow
    distribution across channels.''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Represents the active temperature regulation
    subsystem that maintains mold temperature setpoints. Includes PID tuning parameters
    and mode selection for precise thermal control during molding cycles.'', ''contents'':
    [{''@type'': ''Property'', ''name'': ''setpointTemperature'', ''schema'': ''double'',
    ''description'': ''Desired mold temperature in degrees Celsius.''}, {''@type'':
    ''Property'', ''name'': ''controlMode'', ''schema'': ''string'', ''description'':
    "Operating mode such as ''PID'', ''OnOff'' or ''Adaptive''."}, {''@type'': ''Property'',
    ''name'': ''pidKp'', ''schema'': ''double'', ''description'': ''Proportional gain
    of the PID controller.''}, {''@type'': ''Property'', ''name'': ''pidKi'', ''schema'':
    ''double'', ''description'': ''Integral gain of the PID controller.''}, {''@type'':
    ''Property'', ''name'': ''pidKd'', ''schema'': ''double'', ''description'': ''Derivative
    gain of the PID controller.''}, {''@type'': ''Telemetry'', ''name'': ''actualTemperature'',
    ''schema'': ''double'', ''description'': ''Current measured mold temperature in
    degrees Celsius.''}, {''@type'': ''Telemetry'', ''name'': ''temperatureUniformity'',
    ''schema'': ''double'', ''description'': ''Statistical spread of temperature across
    the mold surface (standard deviation).''}]}'
- source_sentence: I’m looking at the twin ventilation airflow interface for maintaining
    safe underground mine conditions; the device is configured to run in automatic
    mode, its fan speed setpoint is fixed at 78.2, diagnostic monitoring reports a
    fault code of NONE, and the most recent service was performed on 2024‑09‑15 at
    08:30:00 UTC, establishing the baseline for the next scheduled inspection, which
    together ensure predictable operation and clear performance and maintenance indicators.
  sentences:
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Provides predictive analytics and anomaly
    detection on CO₂ trends across the parking facility. The engine stores model versioning,
    processing intervals, and data retention policies to ensure reproducible results.
    Telemetry includes forecasted CO₂ levels, anomaly scores, and actionable recommendations
    for facility managers.'', ''contents'': [{''@type'': ''Property'', ''name'': ''modelVersion'',
    ''schema'': ''string''}, {''@type'': ''Property'', ''name'': ''processingInterval'',
    ''schema'': ''integer'', ''unit'': ''seconds''}, {''@type'': ''Property'', ''name'':
    ''enabled'', ''schema'': ''boolean''}, {''@type'': ''Property'', ''name'': ''dataRetentionDays'',
    ''schema'': ''integer''}, {''@type'': ''Property'', ''name'': ''lastRunTimestamp'',
    ''schema'': ''dateTime''}, {''@type'': ''Telemetry'', ''name'': ''predictedCo2'',
    ''schema'': ''double'', ''unit'': ''ppm''}, {''@type'': ''Telemetry'', ''name'':
    ''anomalyScore'', ''schema'': ''double''}, {''@type'': ''Telemetry'', ''name'':
    ''recommendation'', ''schema'': ''string''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Collects energy usage data for the smart
    window system and calculates savings from tint adjustments. Integrates with home
    energy management to provide real‑time feedback on power consumption. Generates
    alerts when abnormal energy patterns are detected.'', ''contents'': [{''@type'':
    ''Property'', ''name'': ''cumulativeEnergySaved'', ''schema'': ''double'', ''unit'':
    ''kWh'', ''writable'': False, ''description'': ''Total energy saved since installation,
    measured in kilowatt‑hours.''}, {''@type'': ''Property'', ''name'': ''dailyEnergyConsumption'',
    ''schema'': ''double'', ''unit'': ''kWh'', ''writable'': False, ''description'':
    ''Energy consumed by the window system in the last 24‑hour period.''}, {''@type'':
    ''Property'', ''name'': ''peakPowerDemand'', ''schema'': ''double'', ''unit'':
    ''W'', ''writable'': False, ''description'': ''Highest instantaneous power draw
    recorded during a day.''}, {''@type'': ''Property'', ''name'': ''monitoringEnabled'',
    ''schema'': ''boolean'', ''writable'': True, ''description'': ''Enables or disables
    energy monitoring for the device.''}, {''@type'': ''Telemetry'', ''name'': ''energySavedEvent'',
    ''schema'': ''string'', ''description'': "Message indicating a notable energy‑saving
    event, such as ''Tint reduced HVAC load''."}, {''@type'': ''Telemetry'', ''name'':
    ''powerDraw'', ''schema'': ''double'', ''unit'': ''W'', ''description'': ''Current
    power draw of the tinting actuator.''}, {''@type'': ''Telemetry'', ''name'': ''hvacImpact'',
    ''schema'': ''double'', ''unit'': ''W'', ''description'': ''Estimated power reduction
    in the HVAC system due to current tint level.''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Controls the main ventilation fans that
    circulate fresh air throughout the mine shafts. It adjusts fan speed based on
    real‑time airflow demands and can be manually overridden for emergency situations.'',
    ''contents'': [{''@type'': ''Property'', ''name'': ''fanSpeedSetpoint'', ''schema'':
    ''double'', ''description'': ''Desired fan speed as a percentage of maximum output.''},
    {''@type'': ''Property'', ''name'': ''operatingMode'', ''schema'': ''string'',
    ''description'': "Current mode such as ''Automatic'', ''Manual'', or ''Emergency''."},
    {''@type'': ''Property'', ''name'': ''faultCode'', ''schema'': ''string'', ''description'':
    ''Last reported fault code, empty if none.''}, {''@type'': ''Property'', ''name'':
    ''lastMaintenance'', ''schema'': ''string'', ''description'': ''ISO‑8601 date
    of the most recent maintenance activity.''}, {''@type'': ''Telemetry'', ''name'':
    ''fanSpeedActual'', ''schema'': ''double'', ''description'': ''Measured fan speed
    as a percentage of maximum.''}, {''@type'': ''Telemetry'', ''name'': ''powerConsumption'',
    ''schema'': ''double'', ''description'': ''Instantaneous power draw in kilowatts.''},
    {''@type'': ''Telemetry'', ''name'': ''vibrationLevel'', ''schema'': ''double'',
    ''description'': ''Vibration amplitude measured in mm/s RMS.''}, {''@type'': ''Telemetry'',
    ''name'': ''temperature'', ''schema'': ''double'', ''description'': ''Fan housing
    temperature in degrees Celsius.''}]}'
- source_sentence: I’m looking for a digital‑twin interface to support real‑time optimization
    of charging‑station power demand for electric bus fleets that uses the em‑42a
    DC power measurement unit, which operates at a nominal 400 V DC, reports a tariff
    rate of 0.18 $/kWh, and has a measurement accuracy of 0.5 % (ensuring recorded
    values deviate by no more than half a percent), providing reliable, cost‑aware
    measurement for industrial DC power networks and simplifying integration with
    energy‑management systems.
  sentences:
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Tracks electrical energy consumption
    at each charging site for billing and analytics. The meter records cumulative
    energy, instantaneous power, and tariff information. Telemetry streams enable
    precise cost attribution per charging session.'', ''contents'': [{''@type'': ''Property'',
    ''name'': ''meterId'', ''schema'': ''string''}, {''@type'': ''Property'', ''name'':
    ''tariffRate'', ''schema'': ''double''}, {''@type'': ''Property'', ''name'': ''measurementAccuracy'',
    ''schema'': ''double''}, {''@type'': ''Property'', ''name'': ''voltageLevel'',
    ''schema'': ''string''}, {''@type'': ''Telemetry'', ''name'': ''totalEnergyKWh'',
    ''schema'': ''double'', ''unit'': ''kilowattHour''}, {''@type'': ''Telemetry'',
    ''name'': ''instantaneousPowerKW'', ''schema'': ''double'', ''unit'': ''kilowatt''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Executes bilateral energy contracts
    between prosumers based on market signals and individual preferences. It handles
    offer validation, settlement, and real‑time dispatch of energy tokens. The twin
    records each transaction and its state for auditability.'', ''contents'': [{''@type'':
    ''Property'', ''name'': ''maxConcurrentTrades'', ''schema'': ''integer''}, {''@type'':
    ''Property'', ''name'': ''settlementInterval'', ''schema'': ''duration''}, {''@type'':
    ''Property'', ''name'': ''lastSettlementTimestamp'', ''schema'': ''dateTime''},
    {''@type'': ''Property'', ''name'': ''engineStatus'', ''schema'': ''string''},
    {''@type'': ''Telemetry'', ''name'': ''tradeExecuted'', ''schema'': ''object''},
    {''@type'': ''Telemetry'', ''name'': ''tradeFailed'', ''schema'': ''object''},
    {''@type'': ''Telemetry'', ''name'': ''latencyMs'', ''schema'': ''double''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Implements the algorithm that optimizes
    charging load across stations to stay within grid constraints while minimizing
    cost. It stores configuration such as target load and optimization window, and
    emits suggested load set‑points in real time. The twin enables closed‑loop adjustment
    based on observed deviations.'', ''contents'': [{''@type'': ''Property'', ''name'':
    ''optimizerId'', ''schema'': ''string''}, {''@type'': ''Property'', ''name'':
    ''algorithmVersion'', ''schema'': ''string''}, {''@type'': ''Property'', ''name'':
    ''targetLoadKW'', ''schema'': ''double''}, {''@type'': ''Property'', ''name'':
    ''optimizationWindowMin'', ''schema'': ''integer''}, {''@type'': ''Telemetry'',
    ''name'': ''suggestedLoadKW'', ''schema'': ''double'', ''unit'': ''kilowatt''},
    {''@type'': ''Telemetry'', ''name'': ''loadDeviationPercent'', ''schema'': ''double'',
    ''unit'': ''percent''}]}'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on nreimers/MiniLM-L6-H384-uncased

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [nreimers/MiniLM-L6-H384-uncased](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased) on the json dataset. It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [nreimers/MiniLM-L6-H384-uncased](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased) <!-- at revision 3276f0fac9d818781d7a1327b3ff818fc4e643c0 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Training Dataset:**
    - json
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'I’m looking for a digital‑twin interface to support real‑time optimization of charging‑station power demand for electric bus fleets that uses the em‑42a DC power measurement unit, which operates at a nominal 400 V DC, reports a tariff rate of 0.18 $/kWh, and has a measurement accuracy of 0.5 % (ensuring recorded values deviate by no more than half a percent), providing reliable, cost‑aware measurement for industrial DC power networks and simplifying integration with energy‑management systems.',
    "{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Tracks electrical energy consumption at each charging site for billing and analytics. The meter records cumulative energy, instantaneous power, and tariff information. Telemetry streams enable precise cost attribution per charging session.', 'contents': [{'@type': 'Property', 'name': 'meterId', 'schema': 'string'}, {'@type': 'Property', 'name': 'tariffRate', 'schema': 'double'}, {'@type': 'Property', 'name': 'measurementAccuracy', 'schema': 'double'}, {'@type': 'Property', 'name': 'voltageLevel', 'schema': 'string'}, {'@type': 'Telemetry', 'name': 'totalEnergyKWh', 'schema': 'double', 'unit': 'kilowattHour'}, {'@type': 'Telemetry', 'name': 'instantaneousPowerKW', 'schema': 'double', 'unit': 'kilowatt'}]}",
    "{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Implements the algorithm that optimizes charging load across stations to stay within grid constraints while minimizing cost. It stores configuration such as target load and optimization window, and emits suggested load set‑points in real time. The twin enables closed‑loop adjustment based on observed deviations.', 'contents': [{'@type': 'Property', 'name': 'optimizerId', 'schema': 'string'}, {'@type': 'Property', 'name': 'algorithmVersion', 'schema': 'string'}, {'@type': 'Property', 'name': 'targetLoadKW', 'schema': 'double'}, {'@type': 'Property', 'name': 'optimizationWindowMin', 'schema': 'integer'}, {'@type': 'Telemetry', 'name': 'suggestedLoadKW', 'schema': 'double', 'unit': 'kilowatt'}, {'@type': 'Telemetry', 'name': 'loadDeviationPercent', 'schema': 'double', 'unit': 'percent'}]}",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### json

* Dataset: json
* Size: 11,100 training samples
* Columns: <code>query</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                                | positive                                                                              | negative                                                                              |
  |:--------|:-------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                               | string                                                                                | string                                                                                |
  | details | <ul><li>min: 57 tokens</li><li>mean: 119.96 tokens</li><li>max: 200 tokens</li></ul> | <ul><li>min: 239 tokens</li><li>mean: 377.92 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 237 tokens</li><li>mean: 375.91 tokens</li><li>max: 512 tokens</li></ul> |
* Samples:
  | query                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | negative                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>I'm looking at the digital‑twin interface for twin solar irradiance to adjust glass tint dynamically, which maintains a configurable tint level for its display surface; the current targetTintLevel is set to 0.42 indicating a moderate shading preference, adjustments are applied at an adjustmentSpeed of 0.6 to balance responsiveness with stability, the controller operates in an auto operationalMode allowing closed‑loop regulation without manual intervention by continuously comparing actual tint against the target and modifying the output, and the most recent diagnostic check reported a lastErrorCode of E00, denoting normal operation with no faults, ensuring consistent visual performance while providing a clear status indicator for maintenance teams.</code> | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Manages the dynamic tinting of the smart window based on sensor input and user commands. It computes the optimal tint level, drives the electrochromic layer, and reports status and power usage.', 'contents': [{'@type': 'Property', 'name': 'targetTintLevel', 'schema': 'double', 'writable': True, 'description': 'Desired tint level as a percentage from 0 (clear) to 100 (fully tinted).'}, {'@type': 'Property', 'name': 'adjustmentSpeed', 'schema': 'double', 'writable': True, 'description': 'Speed at which the tint is adjusted, expressed in percent per second.'}, {'@type': 'Property', 'name': 'operationalMode', 'schema': 'string', 'writable': True, 'description': "Mode of operation, either 'auto' for sensor‑driven control or 'manual' for user‑directed control."}, {'@type': 'Property', 'name': 'lastErrorCode', 'schema': 'string', 'description': 'Code of the most recent error reported by the...</code> | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Aggregates fleet‑wide statistics and orchestrates robot assignments, monitoring total and active robot counts, average utilization, and overall fleet mode. It provides telemetry for rapid assessment of operational health and detects anomalies across the entire robot population.', 'contents': [{'@type': 'Property', 'name': 'totalRobots', 'schema': 'integer'}, {'@type': 'Property', 'name': 'activeRobots', 'schema': 'integer'}, {'@type': 'Property', 'name': 'averageUtilization', 'schema': 'double'}, {'@type': 'Property', 'name': 'lastUpdate', 'schema': 'string'}, {'@type': 'Property', 'name': 'fleetMode', 'schema': 'string'}, {'@type': 'Telemetry', 'name': 'fleetUtilization', 'schema': 'double'}, {'@type': 'Telemetry', 'name': 'robotHealthSummary', 'schema': 'string'}, {'@type': 'Telemetry', 'name': 'anomalyCount', 'schema': 'integer'}, {'@type': 'Telemetry', 'name': 'fleetModeTelemetry...</code> |
  | <code>The digital‑twin interface for the offshore oil‑rig power model, which models power generation, storage, and consumption on offshore oil rigs, shows that the power generation unit delivers a nominal capacity of 120.5 MW with six turbines collectively producing the rated output, consumes fuel at a rate of 29,800 kg per hour during normal operation, and is currently reported as Running, indicating all systems are active and that this capacity and consumption profile align with the design specifications for mid‑scale thermal facilities while supporting stable grid contribution and meeting efficiency targets.</code>                                                                                                                                                  | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Represents the power generation subsystem on an offshore platform, including gas turbines, diesel generators, and auxiliary renewable sources. It captures static specifications such as rated capacity and dynamic operational data like output power and fuel flow. This model enables monitoring of generation efficiency and availability in harsh marine environments.', 'contents': [{'@type': 'Property', 'name': 'capacityMW', 'schema': 'double'}, {'@type': 'Property', 'name': 'fuelConsumptionRateKgPerHour', 'schema': 'double'}, {'@type': 'Property', 'name': 'turbineCount', 'schema': 'integer'}, {'@type': 'Property', 'name': 'operationalStatus', 'schema': 'string'}, {'@type': 'Telemetry', 'name': 'outputPower', 'schema': 'double'}, {'@type': 'Telemetry', 'name': 'fuelFlow', 'schema': 'double'}, {'@type': 'Telemetry', 'name': 'exhaustTemperatureC', 'schema': 'double'}]}</code>                           | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Continuously monitors vehicle telemetry to detect unsafe conditions and potential collisions. It evaluates risk based on proximity, speed differentials, and environmental factors, and can command emergency braking. The twin reports safety incidents and a computed risk score for each vehicle.', 'contents': [{'@type': 'Property', 'name': 'collisionThreshold', 'schema': 'double'}, {'@type': 'Property', 'name': 'emergencyBrakeEnabled', 'schema': 'boolean'}, {'@type': 'Property', 'name': 'alertLevel', 'schema': 'string'}, {'@type': 'Telemetry', 'name': 'collisionEvents', 'schema': 'integer'}, {'@type': 'Telemetry', 'name': 'brakeActivations', 'schema': 'integer'}, {'@type': 'Telemetry', 'name': 'riskScore', 'schema': 'double'}]}</code>                                                                                                                                                                       |
  | <code>I’m interested in the digital‑twin interface that controls ventilation to maintain safe air quality in parking structures, and I need details about the fan identified by tag fan‑01: it is engineered for a maximum rotational speed of 1500.0 RPM to provide high‑airflow capacity, has an electrical power rating of 2.5 kW to stay within the allocated budget, and is currently in an active state (isOperational = true), so designers can rely on these defined properties for precise HVAC integration, performance modeling, simulations, and maintenance planning.</code>                                                                                                                                                                                                          | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'The Fan Unit twin abstracts an individual ventilation fan installed in the underground parking. It captures mechanical specifications, operational state, and real‑time performance metrics such as speed and power draw. This information feeds the controller to optimize airflow while preventing overloads.', 'contents': [{'@type': 'Property', 'name': 'fanId', 'schema': 'string', 'description': 'Unique identifier for the fan within the system.'}, {'@type': 'Property', 'name': 'maxSpeed', 'schema': 'double', 'description': 'Maximum allowable fan speed as a percentage of full capacity.'}, {'@type': 'Property', 'name': 'powerRating', 'schema': 'double', 'description': 'Rated power consumption at full speed in kilowatts.'}, {'@type': 'Property', 'name': 'isOperational', 'schema': 'boolean', 'description': 'Indicates whether the fan is currently functional.'}, {'@type': 'Telemetry', 'name': 'f...</code> | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'The CO₂ Sensor twin represents a distributed sensor node that continuously measures carbon dioxide levels inside the parking structure. It reports raw concentration values together with environmental context such as temperature, enabling precise ventilation control. The interface also holds calibration and maintenance metadata.', 'contents': [{'@type': 'Property', 'name': 'location', 'schema': 'string', 'description': 'Physical location identifier of the sensor within the parking facility.'}, {'@type': 'Property', 'name': 'measurementInterval', 'schema': 'integer', 'description': 'Time between consecutive measurements in seconds.'}, {'@type': 'Property', 'name': 'calibrationDate', 'schema': 'string', 'description': 'ISO‑8601 date of the last calibration event.'}, {'@type': 'Property', 'name': 'accuracy', 'schema': 'double', 'description': 'Stated measurement accuracy as ± ppm.'}, {'@...</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Evaluation Dataset

#### json

* Dataset: json
* Size: 11,100 evaluation samples
* Columns: <code>query</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                                | positive                                                                              | negative                                                                              |
  |:--------|:-------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                               | string                                                                                | string                                                                                |
  | details | <ul><li>min: 66 tokens</li><li>mean: 119.48 tokens</li><li>max: 217 tokens</li></ul> | <ul><li>min: 221 tokens</li><li>mean: 379.11 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 212 tokens</li><li>mean: 378.49 tokens</li><li>max: 512 tokens</li></ul> |
* Samples:
  | query                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | negative                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Can the digital‑twin interface for detecting and predicting micro‑leakage in high‑pressure hydrogen tanks show me the current firmware version v1.2.3 with its cryptographic hash a3f5c9e8b7d6a1c4e9f2b3d4c5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4, confirm that system diagnostics have cross‑referenced this hash to verify the integrity of the installed binary, and indicate that a newer release exists with the updateAvailable property set to true so I can plan to apply the pending update and keep the device aligned with the latest security patches and feature enhancements?</code>                                                                                                                                                                                                                                                                                                              | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'The twin records the software version running on each sensor component and manages OTA update status. It provides properties for the current firmware hash, update availability, and last update timestamp. Telemetry includes update progress and any error codes during installation.', 'contents': [{'@type': 'Property', 'name': 'firmwareVersion', 'schema': 'string', 'writable': False}, {'@type': 'Property', 'name': 'firmwareHash', 'schema': 'string', 'writable': False}, {'@type': 'Property', 'name': 'updateAvailable', 'schema': 'boolean', 'writable': True}, {'@type': 'Telemetry', 'name': 'updateProgress', 'schema': 'integer', 'unit': '%'}, {'@type': 'Telemetry', 'name': 'lastUpdateTimestamp', 'schema': 'date'}, {'@type': 'Telemetry', 'name': 'errorCode', 'schema': 'integer'}]}</code>                                                                                                                      | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'This twin models the power supply for the sensor suite, tracking battery charge, health, and discharge rates. It raises low‑battery alerts and supports predictive replacement scheduling. The unit also logs power consumption trends for each component.', 'contents': [{'@type': 'Property', 'name': 'batterySerial', 'schema': 'string', 'writable': False}, {'@type': 'Property', 'name': 'capacityWh', 'schema': 'double', 'writable': False}, {'@type': 'Property', 'name': 'lowBatteryThreshold', 'schema': 'double', 'unit': '%', 'writable': True}, {'@type': 'Telemetry', 'name': 'currentCharge', 'schema': 'double', 'unit': '%'}, {'@type': 'Telemetry', 'name': 'voltage', 'schema': 'double', 'unit': 'V'}, {'@type': 'Telemetry', 'name': 'dischargeRate', 'schema': 'double', 'unit': 'W'}]}</code>                                                                                                                      |
  | <code>I’m looking at the digital‑twin interface for optimizing autonomous guided vehicle traffic to reduce bottlenecks in warehouses, and I need to understand how its current settings will affect performance: the twin is set with a sensitivityLevel of 3, which gives it a moderate threshold for spotting relevant changes in the environment and determines how aggressively it compares incoming data to expected behavior; its monitoringWindowSec is 120, so it aggregates and evaluates state information over a two‑minute period to smooth out transient fluctuations while still staying responsive; and the active flag is true, meaning the twin is presently running its monitoring and control loops and continuously applying the defined sensitivity and window to keep an up‑to‑date model of the physical assets, providing engineers with a predictable baseline for tuning and testing.</code> | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Monitors traffic streams to identify zones where AGV flow is constrained, creating potential delays. The detector uses configurable sensitivity and time windows to differentiate transient spikes from persistent bottlenecks. Its twin exposes detection results and health status for upstream optimization components.', 'contents': [{'@type': 'Property', 'name': 'sensitivityLevel', 'schema': 'integer', 'writable': True, 'description': 'Scale from 1 (low) to 5 (high) that adjusts how aggressively the detector flags bottlenecks.'}, {'@type': 'Property', 'name': 'monitoringWindowSec', 'schema': 'integer', 'writable': True, 'description': 'Length of the time window, in seconds, over which traffic metrics are aggregated.'}, {'@type': 'Property', 'name': 'active', 'schema': 'boolean', 'writable': True, 'description': 'Indicates whether the detector is currently monitoring traffic.'}, {'@type': ...</code> | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Represents an individual autonomous guided vehicle participating in the swarm. The twin captures static capabilities such as payload and speed limits as well as dynamic state like battery level and assigned task. It enables the system to monitor each AGV and adjust its behavior in real time.', 'contents': [{'@type': 'Property', 'name': 'agvId', 'schema': 'string', 'writable': False, 'description': 'Unique identifier of the AGV within the fleet.'}, {'@type': 'Property', 'name': 'maxSpeed', 'schema': 'double', 'writable': True, 'unit': 'm/s', 'description': 'Maximum allowed speed for the vehicle.'}, {'@type': 'Property', 'name': 'payloadCapacityKg', 'schema': 'double', 'writable': True, 'unit': 'kg', 'description': 'Maximum payload the AGV can carry.'}, {'@type': 'Property', 'name': 'batteryLevelPct', 'schema': 'double', 'writable': False, 'unit': 'percent', 'description': 'Current bat...</code> |
  | <code>I’m looking for a digital‑twin interface for the twin of crack growth in wind turbine blades that samples at 5 samples per second, is mounted on the blade‑inner‑surface to monitor the critical aerodynamic region, and delivers a measurement accuracy of 0.05, providing high‑fidelity monitoring of blade performance under varying conditions, detecting transient events that could affect structural integrity while reducing exposure to external contaminants and supporting predictive‑maintenance algorithms.</code>                                                                                                                                                                                                                                                                                                                                                                                  | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Monitors blade and ambient temperatures which influence material fatigue properties. Sensors are embedded within the composite layers to capture internal thermal gradients. Temperature data feeds the stress‑temperature coupling in the fatigue model.', 'contents': [{'@type': 'Property', 'name': 'samplingRate', 'schema': 'integer', 'unit': 'Hz', 'description': 'Rate at which temperature samples are taken'}, {'@type': 'Property', 'name': 'sensorLocation', 'schema': 'string', 'description': 'Location of the sensor relative to the blade surface'}, {'@type': 'Property', 'name': 'measurementAccuracy', 'schema': 'double', 'unit': '°C', 'description': 'Maximum error of temperature readings'}, {'@type': 'Telemetry', 'name': 'bladeTemperature', 'schema': 'double', 'unit': '°C', 'description': 'Internal temperature of the blade material'}, {'@type': 'Telemetry', 'name': 'ambientTemperature', 'sc...</code> | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Captures high‑frequency blade vibrations to identify dynamic loading conditions that accelerate fatigue. The sensor is mounted near the root where strain concentrations are highest. Data is streamed in real time for integration with the crack growth model.', 'contents': [{'@type': 'Property', 'name': 'samplingRate', 'schema': 'integer', 'unit': 'Hz', 'description': 'Number of vibration samples captured per second'}, {'@type': 'Property', 'name': 'sensorLocation', 'schema': 'string', 'description': 'Physical mounting point on the blade'}, {'@type': 'Property', 'name': 'sensitivity', 'schema': 'double', 'unit': 'dB/g', 'description': 'Sensor sensitivity to acceleration'}, {'@type': 'Telemetry', 'name': 'vibrationAmplitude', 'schema': 'double', 'unit': 'mm/s', 'description': 'Peak vibration amplitude'}, {'@type': 'Telemetry', 'name': 'dominantFrequency', 'schema': 'double', 'unit': 'Hz'...</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 2
- `per_device_eval_batch_size`: 2
- `learning_rate`: 2e-05
- `num_train_epochs`: 5
- `warmup_ratio`: 0.1
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 2
- `per_device_eval_batch_size`: 2
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 5
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0901 | 100  | -             | 2.6889          |
| 0.1802 | 200  | -             | 1.2904          |
| 0.2703 | 300  | -             | 0.6380          |
| 0.3604 | 400  | -             | 0.4231          |
| 0.4505 | 500  | 1.4145        | 0.3351          |
| 0.5405 | 600  | -             | 0.2625          |
| 0.6306 | 700  | -             | 0.2369          |
| 0.7207 | 800  | -             | 0.1811          |
| 0.8108 | 900  | -             | 0.1525          |
| 0.9009 | 1000 | 0.2604        | 0.1377          |
| 0.9910 | 1100 | -             | 0.1390          |
| 1.0811 | 1200 | -             | 0.1210          |
| 1.1712 | 1300 | -             | 0.1279          |
| 1.2613 | 1400 | -             | 0.0915          |
| 1.3514 | 1500 | 0.1364        | 0.0887          |
| 1.4414 | 1600 | -             | 0.0775          |
| 1.5315 | 1700 | -             | 0.0737          |
| 1.6216 | 1800 | -             | 0.0803          |
| 1.7117 | 1900 | -             | 0.0774          |
| 1.8018 | 2000 | 0.0676        | 0.0724          |
| 1.8919 | 2100 | -             | 0.0702          |
| 1.9820 | 2200 | -             | 0.0666          |
| 2.0721 | 2300 | -             | 0.0595          |
| 2.1622 | 2400 | -             | 0.0678          |
| 2.2523 | 2500 | 0.0416        | 0.0616          |
| 2.3423 | 2600 | -             | 0.0545          |
| 2.4324 | 2700 | -             | 0.0524          |
| 2.5225 | 2800 | -             | 0.0508          |
| 2.6126 | 2900 | -             | 0.0594          |
| 2.7027 | 3000 | 0.0259        | 0.0578          |
| 2.7928 | 3100 | -             | 0.0557          |
| 2.8829 | 3200 | -             | 0.0552          |
| 2.9730 | 3300 | -             | 0.0549          |
| 3.0631 | 3400 | -             | 0.0536          |
| 3.1532 | 3500 | 0.0193        | 0.0523          |
| 3.2432 | 3600 | -             | 0.0541          |
| 3.3333 | 3700 | -             | 0.0493          |
| 3.4234 | 3800 | -             | 0.0500          |
| 3.5135 | 3900 | -             | 0.0499          |
| 3.6036 | 4000 | 0.0154        | 0.0470          |
| 3.6937 | 4100 | -             | 0.0519          |
| 3.7838 | 4200 | -             | 0.0536          |
| 3.8739 | 4300 | -             | 0.0507          |
| 3.9640 | 4400 | -             | 0.0486          |
| 4.0541 | 4500 | 0.0103        | 0.0499          |
| 4.1441 | 4600 | -             | 0.0452          |
| 4.2342 | 4700 | -             | 0.0481          |
| 4.3243 | 4800 | -             | 0.0469          |
| 4.4144 | 4900 | -             | 0.0465          |
| 4.5045 | 5000 | 0.0094        | 0.0446          |
| 4.5946 | 5100 | -             | 0.0456          |
| 4.6847 | 5200 | -             | 0.0481          |
| 4.7748 | 5300 | -             | 0.0475          |
| 4.8649 | 5400 | -             | 0.0472          |
| 4.9550 | 5500 | 0.0086        | 0.0469          |


### Framework Versions
- Python: 3.10.12
- Sentence Transformers: 3.3.1
- Transformers: 4.57.1
- PyTorch: 2.8.0+cu128
- Accelerate: 1.11.0
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->