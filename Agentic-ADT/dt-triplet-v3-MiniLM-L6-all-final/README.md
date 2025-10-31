---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:24747
- loss:MultipleNegativesRankingLoss
base_model: nreimers/MiniLM-L6-H384-uncased
widget:
- source_sentence: What is the best suitable digital twin interface for Battery Management?
    Only give the interface.
  sentences:
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Orchestrates simulation runs, schedules
    updates, and exposes control commands for the urban noise attenuation twin. It
    tracks simulation status, iteration count, and provides hooks for external triggers.
    This component enables automated scenario testing and integration with city planning
    workflows.'', ''contents'': [{''@type'': ''Property'', ''name'': ''simulationId'',
    ''schema'': ''string''}, {''@type'': ''Property'', ''name'': ''status'', ''schema'':
    ''string''}, {''@type'': ''Property'', ''name'': ''currentIteration'', ''schema'':
    ''integer''}, {''@type'': ''Property'', ''name'': ''totalIterations'', ''schema'':
    ''integer''}, {''@type'': ''Telemetry'', ''name'': ''iterationCompleted'', ''schema'':
    ''integer''}, {''@type'': ''Telemetry'', ''name'': ''iterationTimestamp'', ''schema'':
    ''long''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Encapsulates the AI model that predicts
    acoustic comfort scores based on sensor inputs and occupancy data. It maintains
    model versioning, threshold parameters, and policy settings for comfort evaluation.
    The twin can be updated remotely for continuous learning.'', ''contents'': [{''@type'':
    ''Property'', ''name'': ''modelVersion'', ''schema'': ''string''}, {''@type'':
    ''Property'', ''name'': ''comfortThreshold'', ''schema'': ''double''}, {''@type'':
    ''Property'', ''name'': ''occupiedStatus'', ''schema'': ''boolean''}, {''@type'':
    ''Property'', ''name'': ''acousticPolicy'', ''schema'': ''string''}, {''@type'':
    ''Telemetry'', ''name'': ''comfortScore'', ''schema'': ''double'', ''unit'': ''%''},
    {''@type'': ''Telemetry'', ''name'': ''predictedNoise'', ''schema'': ''double'',
    ''unit'': ''dB''}, {''@type'': ''Telemetry'', ''name'': ''predictionLatency'',
    ''schema'': ''integer'', ''unit'': ''ms''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Represents the battery energy storage
    system responsible for absorbing excess renewable generation and delivering power
    on demand. It tracks capacity, state of charge, health, and operational limits
    to enable optimal dispatch decisions. The model also exposes runtime metrics for
    power flow and temperature monitoring.'', ''contents'': [{''@type'': ''Property'',
    ''name'': ''capacityKWh'', ''schema'': ''double'', ''writable'': True}, {''@type'':
    ''Property'', ''name'': ''stateOfChargePct'', ''schema'': ''double'', ''writable'':
    True}, {''@type'': ''Property'', ''name'': ''maxChargeRateKW'', ''schema'': ''double''},
    {''@type'': ''Property'', ''name'': ''minDischargeRateKW'', ''schema'': ''double''},
    {''@type'': ''Telemetry'', ''name'': ''currentPowerKW'', ''schema'': ''double'',
    ''unit'': ''kW''}, {''@type'': ''Telemetry'', ''name'': ''batteryTemperatureC'',
    ''schema'': ''double'', ''unit'': ''C''}]}'
- source_sentence: What is the best suitable digital twin interface for EnergyStorage?
    Only give the interface.
  sentences:
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''The EnergyStorage component models on‑board
    batteries or super‑capacitors that buffer regenerative energy and supply auxiliary
    loads. It tracks state‑of‑charge, health, temperature, and charge/discharge limits
    to prevent over‑stress during aggressive optimization. This twin enables the optimizer
    to plan energy exchanges with the storage system.'', ''contents'': [{''@type'':
    ''Property'', ''name'': ''stateOfCharge'', ''schema'': ''double'', ''unit'': ''percent'',
    ''writable'': False}, {''@type'': ''Property'', ''name'': ''maxChargePower'',
    ''schema'': ''double'', ''unit'': ''kW'', ''writable'': True}, {''@type'': ''Property'',
    ''name'': ''maxDischargePower'', ''schema'': ''double'', ''unit'': ''kW'', ''writable'':
    True}, {''@type'': ''Property'', ''name'': ''temperature'', ''schema'': ''double'',
    ''unit'': ''C'', ''writable'': False}, {''@type'': ''Telemetry'', ''name'': ''chargePower'',
    ''schema'': ''double'', ''unit'': ''kW''}, {''@type'': ''Telemetry'', ''name'':
    ''dischargePower'', ''schema'': ''double'', ''unit'': ''kW''}, {''@type'': ''Telemetry'',
    ''name'': ''healthIndex'', ''schema'': ''double'', ''unit'': ''percent''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Provides the ambient oceanographic context
    surrounding the cable, such as depth, seabed type, and currents. These conditions
    are crucial inputs for fatigue loading calculations. The component updates in
    near‑real‑time from ocean monitoring services.'', ''contents'': [{''@type'': ''Property'',
    ''name'': ''waterDepth'', ''schema'': ''double'', ''unit'': ''meter'', ''description'':
    ''Mean depth of the water column above the cable.''}, {''@type'': ''Property'',
    ''name'': ''seabedType'', ''schema'': ''string'', ''description'': ''Classification
    of the seabed material (e.g., sand, silt, rock).''}, {''@type'': ''Property'',
    ''name'': ''currentSpeed'', ''schema'': ''double'', ''unit'': ''meter/second'',
    ''description'': ''Average near‑bed current velocity.''}, {''@type'': ''Property'',
    ''name'': ''temperatureProfile'', ''schema'': ''string'', ''description'': ''Encoded
    profile of temperature variation with depth.''}, {''@type'': ''Telemetry'', ''name'':
    ''ambientTemperature'', ''schema'': ''double'', ''unit'': ''celsius'', ''description'':
    ''Current water temperature at cable depth.''}, {''@type'': ''Telemetry'', ''name'':
    ''hydrostaticPressure'', ''schema'': ''double'', ''unit'': ''bar'', ''description'':
    ''Pressure exerted by the water column at the cable location.''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''The TractionController twin represents
    the low‑level control logic that translates power set‑points into motor torque
    commands for each axle. It monitors wheel slip, adhesion limits, and regenerative
    braking capacity, ensuring safe and efficient traction under varying track conditions.
    The model captures configuration parameters, real‑time state, and diagnostic events.'',
    ''contents'': [{''@type'': ''Property'', ''name'': ''maxTorque'', ''schema'':
    ''double'', ''unit'': ''Nm'', ''writable'': True}, {''@type'': ''Property'', ''name'':
    ''wheelSlipThreshold'', ''schema'': ''double'', ''unit'': ''percent'', ''writable'':
    True}, {''@type'': ''Property'', ''name'': ''regenerativeBrakingEnabled'', ''schema'':
    ''boolean'', ''writable'': True}, {''@type'': ''Telemetry'', ''name'': ''actualTorque'',
    ''schema'': ''double'', ''unit'': ''Nm''}, {''@type'': ''Telemetry'', ''name'':
    ''wheelSlip'', ''schema'': ''double'', ''unit'': ''percent''}, {''@type'': ''Telemetry'',
    ''name'': ''brakeEnergyRecovered'', ''schema'': ''double'', ''unit'': ''kWh''}]}'
- source_sentence: What is the best suitable digital twin interface for MaintenanceScheduler?
    Only give the interface.
  sentences:
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Controls the speed of HVAC supply and
    return fans in each mechanical room. The twin tracks current RPM, motor health,
    and allows remote set‑point adjustments to balance airflow and energy use. Integrated
    diagnostics help detect bearing wear or voltage anomalies.'', ''contents'': [{''@type'':
    ''Property'', ''name'': ''fanId'', ''schema'': ''string'', ''writable'': False},
    {''@type'': ''Property'', ''name'': ''desiredSpeedRpm'', ''schema'': ''integer'',
    ''writable'': True}, {''@type'': ''Property'', ''name'': ''maxSpeedRpm'', ''schema'':
    ''integer'', ''writable'': False}, {''@type'': ''Property'', ''name'': ''motorHealth'',
    ''schema'': ''string'', ''writable'': False}, {''@type'': ''Telemetry'', ''name'':
    ''actualSpeedRpm'', ''schema'': ''integer''}, {''@type'': ''Telemetry'', ''name'':
    ''powerConsumptionW'', ''schema'': ''double''}, {''@type'': ''Telemetry'', ''name'':
    ''vibrationLevel'', ''schema'': ''double''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Collects real‑time oceanographic data
    that drive fatigue calculations. Sensors measure current velocity, temperature,
    and pressure at the cable location. The twin aggregates these readings for downstream
    analytics.'', ''contents'': [{''@type'': ''Property'', ''name'': ''sensorId'',
    ''schema'': ''string''}, {''@type'': ''Property'', ''name'': ''deploymentDepth'',
    ''schema'': ''double'', ''unit'': ''m''}, {''@type'': ''Property'', ''name'':
    ''locationLatitude'', ''schema'': ''double''}, {''@type'': ''Property'', ''name'':
    ''locationLongitude'', ''schema'': ''double''}, {''@type'': ''Telemetry'', ''name'':
    ''currentVelocity'', ''schema'': ''double'', ''unit'': ''m/s''}, {''@type'': ''Telemetry'',
    ''name'': ''waterTemperature'', ''schema'': ''double'', ''unit'': ''°C''}, {''@type'':
    ''Telemetry'', ''name'': ''hydrostaticPressure'', ''schema'': ''double'', ''unit'':
    ''kPa''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Plans and records maintenance activities
    based on fatigue predictions and risk thresholds. It stores scheduled inspection
    dates, crew assignments, and status of completed work. The twin can trigger alerts
    when a segment approaches a critical fatigue level.'', ''contents'': [{''@type'':
    ''Property'', ''name'': ''nextInspectionDate'', ''schema'': ''date''}, {''@type'':
    ''Property'', ''name'': ''inspectionIntervalDays'', ''schema'': ''integer''},
    {''@type'': ''Property'', ''name'': ''assignedCrew'', ''schema'': ''string''},
    {''@type'': ''Property'', ''name'': ''lastMaintenanceStatus'', ''schema'': ''string''},
    {''@type'': ''Telemetry'', ''name'': ''inspectionDueInDays'', ''schema'': ''integer''},
    {''@type'': ''Telemetry'', ''name'': ''maintenanceCostEstimate'', ''schema'':
    ''double'', ''unit'': ''USD''}]}'
- source_sentence: What is the best suitable digital twin interface for Control Logic?
    Only give the interface.
  sentences:
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Aggregates stored, charged, and discharged
    thermal energy for the stratified storage system. Computes net energy efficiency
    and tracks cumulative energy throughput over time. Provides key performance indicators
    for plant operators.'', ''contents'': [{''@type'': ''Property'', ''name'': ''totalStoredEnergy'',
    ''schema'': ''double'', ''description'': ''Current total thermal energy stored
    in the tank in megajoules.'', ''unit'': ''MJ''}, {''@type'': ''Property'', ''name'':
    ''cumulativeChargeEnergy'', ''schema'': ''double'', ''description'': ''Cumulative
    energy added to the tank since commissioning in megajoules.'', ''unit'': ''MJ''},
    {''@type'': ''Telemetry'', ''name'': ''chargeRate'', ''schema'': ''double'', ''description'':
    ''Instantaneous charging power in megawatts.'', ''unit'': ''MW''}, {''@type'':
    ''Telemetry'', ''name'': ''dischargeRate'', ''schema'': ''double'', ''description'':
    ''Instantaneous discharging power in megawatts.'', ''unit'': ''MW''}, {''@type'':
    ''Telemetry'', ''name'': ''roundTripEfficiency'', ''schema'': ''double'', ''description'':
    ''Measured round‑trip thermal energy efficiency as a percentage.'', ''unit'':
    ''%''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Represents a sensor node that measures
    volumetric water content in the soil. It reports moisture, temperature, and electrical
    conductivity to enable precise irrigation decisions. The twin captures calibration
    data and deployment location for accurate modeling.'', ''contents'': [{''@type'':
    ''Property'', ''name'': ''sensorId'', ''schema'': ''string''}, {''@type'': ''Property'',
    ''name'': ''location'', ''schema'': ''string''}, {''@type'': ''Property'', ''name'':
    ''measurementInterval'', ''schema'': ''integer'', ''description'': ''Interval
    in seconds between successive readings''}, {''@type'': ''Property'', ''name'':
    ''calibrationDate'', ''schema'': ''date''}, {''@type'': ''Telemetry'', ''name'':
    ''moistureLevel'', ''schema'': ''double'', ''description'': ''Volumetric water
    content as a percentage'', ''unit'': ''%''}, {''@type'': ''Telemetry'', ''name'':
    ''soilTemperature'', ''schema'': ''double'', ''unit'': ''°C''}, {''@type'': ''Telemetry'',
    ''name'': ''electricalConductivity'', ''schema'': ''double'', ''description'':
    ''Soil EC in dS/m'', ''unit'': ''dS/m''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Encapsulates the algorithmic rules that
    maintain stratification while meeting demand. Adjusts pump speed, valve positions,
    and setpoints based on temperature gradients and forecasted loads. Emits alerts
    when control actions approach operational limits.'', ''contents'': [{''@type'':
    ''Property'', ''name'': ''targetStratification'', ''schema'': ''string'', ''description'':
    ''Desired stratification profile (e.g., "hot‑top, cold‑bottom").''}, {''@type'':
    ''Property'', ''name'': ''maxPumpSpeed'', ''schema'': ''double'', ''description'':
    ''Maximum allowable pump speed in rpm for control actions.'', ''unit'': ''rpm''},
    {''@type'': ''Telemetry'', ''name'': ''controlAction'', ''schema'': ''string'',
    ''description'': ''Description of the most recent control adjustment (e.g., "increase
    pump speed by 5%" ).''}, {''@type'': ''Telemetry'', ''name'': ''alertLevel'',
    ''schema'': ''string'', ''description'': ''Current alert severity (none, warning,
    critical).''}]}'
- source_sentence: What is the best suitable digital twin interface for MaintenanceSchedule?
    Only give the interface.
  sentences:
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Controls the fresh‑air supply for a
    school HVAC zone based on occupancy and air‑quality inputs. It calculates a target
    ventilation rate and adjusts fan speed to meet that rate while respecting energy
    constraints. The twin records adjustments and operational status for performance
    monitoring.'', ''contents'': [{''@type'': ''Property'', ''name'': ''zoneId'',
    ''schema'': ''string''}, {''@type'': ''Property'', ''name'': ''targetVentilationRate'',
    ''schema'': ''double''}, {''@type'': ''Property'', ''name'': ''fanSpeedSetpoint'',
    ''schema'': ''integer''}, {''@type'': ''Property'', ''name'': ''controlMode'',
    ''schema'': ''string''}, {''@type'': ''Property'', ''name'': ''lastAdjustment'',
    ''schema'': ''dateTime''}, {''@type'': ''Telemetry'', ''name'': ''currentVentilationRate'',
    ''schema'': ''double''}, {''@type'': ''Telemetry'', ''name'': ''fanSpeedActual'',
    ''schema'': ''integer''}, {''@type'': ''Telemetry'', ''name'': ''powerConsumption'',
    ''schema'': ''double''}, {''@type'': ''Telemetry'', ''name'': ''controlModeActive'',
    ''schema'': ''string''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Encapsulates the analytical model that
    predicts coating wear and its impact on photovoltaic efficiency. It stores calibration
    parameters, versioning, and the latest computed efficiency loss forecast. The
    model updates continuously based on incoming condition and exposure data.'', ''contents'':
    [{''@type'': ''Property'', ''name'': ''modelVersion'', ''schema'': ''string''},
    {''@type'': ''Property'', ''name'': ''lastCalibrationDate'', ''schema'': ''dateTime''},
    {''@type'': ''Property'', ''name'': ''degradationCoefficient'', ''schema'': ''double''},
    {''@type'': ''Property'', ''name'': ''predictedEfficiencyLossPercent'', ''schema'':
    ''double'', ''unit'': ''percent''}, {''@type'': ''Telemetry'', ''name'': ''predictedEfficiencyPercent'',
    ''schema'': ''double'', ''unit'': ''percent''}, {''@type'': ''Telemetry'', ''name'':
    ''predictedRemainingLifetimeYears'', ''schema'': ''double'', ''unit'': ''year''},
    {''@type'': ''Telemetry'', ''name'': ''modelUpdateTimestamp'', ''schema'': ''dateTime''}]}'
  - '{''@context'': ''dtmi:dtdl:context;2'', ''@id'': '''', ''@type'': ''Interface'',
    ''displayName'': '''', ''description'': ''Tracks scheduled and performed maintenance
    actions intended to restore or preserve coating performance, such as cleaning
    or recoating. It records dates, frequencies, and status flags to support operational
    planning and compliance reporting. This component links maintenance events to
    observed changes in coating telemetry.'', ''contents'': [{''@type'': ''Property'',
    ''name'': ''lastCleaningDate'', ''schema'': ''dateTime''}, {''@type'': ''Property'',
    ''name'': ''cleaningFrequencyDays'', ''schema'': ''integer''}, {''@type'': ''Property'',
    ''name'': ''nextInspectionDate'', ''schema'': ''dateTime''}, {''@type'': ''Property'',
    ''name'': ''maintenanceStatus'', ''schema'': ''string''}, {''@type'': ''Telemetry'',
    ''name'': ''cleaningPerformed'', ''schema'': ''boolean''}, {''@type'': ''Telemetry'',
    ''name'': ''inspectionResult'', ''schema'': ''string''}, {''@type'': ''Telemetry'',
    ''name'': ''maintenanceActionTimestamp'', ''schema'': ''dateTime''}]}'
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
    'What is the best suitable digital twin interface for MaintenanceSchedule? Only give the interface.',
    "{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Tracks scheduled and performed maintenance actions intended to restore or preserve coating performance, such as cleaning or recoating. It records dates, frequencies, and status flags to support operational planning and compliance reporting. This component links maintenance events to observed changes in coating telemetry.', 'contents': [{'@type': 'Property', 'name': 'lastCleaningDate', 'schema': 'dateTime'}, {'@type': 'Property', 'name': 'cleaningFrequencyDays', 'schema': 'integer'}, {'@type': 'Property', 'name': 'nextInspectionDate', 'schema': 'dateTime'}, {'@type': 'Property', 'name': 'maintenanceStatus', 'schema': 'string'}, {'@type': 'Telemetry', 'name': 'cleaningPerformed', 'schema': 'boolean'}, {'@type': 'Telemetry', 'name': 'inspectionResult', 'schema': 'string'}, {'@type': 'Telemetry', 'name': 'maintenanceActionTimestamp', 'schema': 'dateTime'}]}",
    "{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Encapsulates the analytical model that predicts coating wear and its impact on photovoltaic efficiency. It stores calibration parameters, versioning, and the latest computed efficiency loss forecast. The model updates continuously based on incoming condition and exposure data.', 'contents': [{'@type': 'Property', 'name': 'modelVersion', 'schema': 'string'}, {'@type': 'Property', 'name': 'lastCalibrationDate', 'schema': 'dateTime'}, {'@type': 'Property', 'name': 'degradationCoefficient', 'schema': 'double'}, {'@type': 'Property', 'name': 'predictedEfficiencyLossPercent', 'schema': 'double', 'unit': 'percent'}, {'@type': 'Telemetry', 'name': 'predictedEfficiencyPercent', 'schema': 'double', 'unit': 'percent'}, {'@type': 'Telemetry', 'name': 'predictedRemainingLifetimeYears', 'schema': 'double', 'unit': 'year'}, {'@type': 'Telemetry', 'name': 'modelUpdateTimestamp', 'schema': 'dateTime'}]}",
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
* Size: 24,747 training samples
* Columns: <code>query</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                              | positive                                                                              | negative                                                                              |
  |:--------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                                | string                                                                                |
  | details | <ul><li>min: 18 tokens</li><li>mean: 20.54 tokens</li><li>max: 25 tokens</li></ul> | <ul><li>min: 238 tokens</li><li>mean: 373.22 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 227 tokens</li><li>mean: 374.71 tokens</li><li>max: 512 tokens</li></ul> |
* Samples:
  | query                                                                                                                 | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | negative                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  |:----------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What is the best suitable digital twin interface for Maintenance Scheduler? Only give the interface.</code>     | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'The Maintenance Scheduler component orchestrates service windows based on wear predictions and operational constraints. It issues alerts, tracks upcoming maintenance dates, and can toggle maintenance modes to pause or continue production. Integration with plant ERP ensures work orders are generated automatically.', 'contents': [{'@type': 'Property', 'name': 'maintenanceWindowHours', 'schema': 'integer'}, {'@type': 'Property', 'name': 'nextMaintenanceDue', 'schema': 'string'}, {'@type': 'Property', 'name': 'maintenanceMode', 'schema': 'string'}, {'@type': 'Property', 'name': 'alertEnabled', 'schema': 'boolean'}, {'@type': 'Telemetry', 'name': 'maintenanceAlert', 'schema': 'string'}, {'@type': 'Telemetry', 'name': 'maintenanceProgress', 'schema': 'double'}]}</code>                                                                                                                                | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'The Joint Health Model component runs a predictive algorithm that estimates future wear based on historical sensor streams and operating conditions. It provides risk scores and projected wear rates to enable proactive part replacement before failure. Model updates are versioned and retrained periodically with new data.', 'contents': [{'@type': 'Property', 'name': 'modelVersion', 'schema': 'string'}, {'@type': 'Property', 'name': 'predictionHorizonMinutes', 'schema': 'integer'}, {'@type': 'Property', 'name': 'confidenceThreshold', 'schema': 'double'}, {'@type': 'Property', 'name': 'lastTrainingDate', 'schema': 'string'}, {'@type': 'Telemetry', 'name': 'predictedWearRate', 'schema': 'double'}, {'@type': 'Telemetry', 'name': 'wearRiskScore', 'schema': 'double'}, {'@type': 'Telemetry', 'name': 'modelAccuracy', 'schema': 'double'}]}</code>                                                             |
  | <code>What is the best suitable digital twin interface for DriverAssist? Only give the interface.</code>              | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Represents advanced driver‑assist features that influence vehicle speed and spacing, such as adaptive cruise control and lane‑keeping. Telemetry provides the actual following distance and lane offset used by the control algorithms. This twin helps assess how assist systems modify fuel consumption patterns.', 'contents': [{'@type': 'Property', 'name': 'adaptiveCruiseEnabled', 'schema': 'boolean'}, {'@type': 'Property', 'name': 'laneKeepingEnabled', 'schema': 'boolean'}, {'@type': 'Property', 'name': 'maxAssistSpeed', 'schema': 'double', 'unit': 'km/h'}, {'@type': 'Telemetry', 'name': 'followingDistance', 'schema': 'double', 'unit': 'm'}, {'@type': 'Telemetry', 'name': 'laneOffset', 'schema': 'double', 'unit': 'm'}]}</code>                                                                                                                                                                          | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Captures the type of drivetrain and its gear characteristics, essential for translating engine output to wheel torque. Real‑time telemetry includes current gear and torque‑converter slip. This component enables accurate estimation of mechanical losses affecting fuel usage.', 'contents': [{'@type': 'Property', 'name': 'transmissionType', 'schema': 'string'}, {'@type': 'Property', 'name': 'gearCount', 'schema': 'integer'}, {'@type': 'Property', 'name': 'finalDriveRatio', 'schema': 'double'}, {'@type': 'Telemetry', 'name': 'gearPosition', 'schema': 'string'}, {'@type': 'Telemetry', 'name': 'torqueConverterSlip', 'schema': 'double', 'unit': '%'}]}</code>                                                                                                                                                                                                                                                         |
  | <code>What is the best suitable digital twin interface for DistrictTemperatureSensor? Only give the interface.</code> | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Represents a temperature sensor deployed within a city district to capture fine-grained thermal data. The twin records geographic placement, operational status, and measurement cadence to support spatial heat mapping. It enables continuous streaming of temperature-related telemetry for analysis.', 'contents': [{'@type': 'Property', 'name': 'districtId', 'schema': 'string'}, {'@type': 'Property', 'name': 'latitude', 'schema': 'double'}, {'@type': 'Property', 'name': 'longitude', 'schema': 'double'}, {'@type': 'Property', 'name': 'sensorStatus', 'schema': 'string'}, {'@type': 'Property', 'name': 'measurementInterval', 'schema': 'integer'}, {'@type': 'Telemetry', 'name': 'temperature', 'schema': 'double', 'unit': 'Celsius'}, {'@type': 'Telemetry', 'name': 'humidity', 'schema': 'double', 'unit': '%'}, {'@type': 'Telemetry', 'name': 'heatIndex', 'schema': 'double', 'unit': 'Celsius'}]}</code> | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Evaluates the potential impact of the fire on human life, infrastructure, and ecological assets. It combines predicted fire behavior with population density, asset locations, and evacuation routes to generate risk scores. The model updates risk metrics as the fire evolves and can trigger alerts for high‑risk zones.', 'contents': [{'@type': 'Property', 'name': 'populationDensityGrid', 'schema': 'string', 'writable': True}, {'@type': 'Property', 'name': 'criticalInfrastructureList', 'schema': 'string[]', 'writable': True}, {'@type': 'Property', 'name': 'evacuationRouteStatus', 'schema': 'string', 'writable': True}, {'@type': 'Property', 'name': 'riskThreshold', 'schema': 'double', 'unit': '', 'writable': True}, {'@type': 'Telemetry', 'name': 'riskScoreMap', 'schema': 'string', 'description': 'Raster of computed risk scores per grid cell'}, {'@type': 'Telemetry', 'name': 'highRiskAlert'...</code> |
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
* Size: 24,747 evaluation samples
* Columns: <code>query</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                              | positive                                                                              | negative                                                                              |
  |:--------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                                | string                                                                                |
  | details | <ul><li>min: 18 tokens</li><li>mean: 20.53 tokens</li><li>max: 26 tokens</li></ul> | <ul><li>min: 227 tokens</li><li>mean: 377.77 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 236 tokens</li><li>mean: 372.64 tokens</li><li>max: 512 tokens</li></ul> |
* Samples:
  | query                                                                                                             | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | negative                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  |:------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What is the best suitable digital twin interface for Thermal Rating Engine? Only give the interface.</code> | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Calculates the real‑time thermal capacity of the substation based on equipment temperatures and load forecasts. It applies a validated heat balance algorithm and updates the thermal margin continuously. The engine runs in a containerized service for reliable deployment.', 'contents': [{'@type': 'Property', 'name': 'algorithmVersion', 'schema': 'string'}, {'@type': 'Property', 'name': 'maxThermalCapacityMW', 'schema': 'double'}, {'@type': 'Property', 'name': 'lastCalibration', 'schema': 'dateTime'}, {'@type': 'Telemetry', 'name': 'currentThermalLoadMW', 'schema': 'double'}, {'@type': 'Telemetry', 'name': 'thermalMarginPercent', 'schema': 'double'}]}</code>                                                                                                                                     | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Manages scheduled maintenance activities based on degradation forecasts and real‑time alerts. It calculates upcoming service windows, tracks completed interventions, and triggers alerts when thresholds are exceeded. This component helps ensure optimal panel performance and longevity.', 'contents': [{'@type': 'Property', 'name': 'maintenanceInterval', 'schema': 'integer', 'description': 'Standard interval between maintenance actions in days.', 'unit': 'day'}, {'@type': 'Property', 'name': 'lastMaintenanceDate', 'schema': 'dateTime', 'description': 'Timestamp of the most recent maintenance activity.'}, {'@type': 'Property', 'name': 'nextMaintenanceDue', 'schema': 'dateTime', 'description': 'Calculated date when the next maintenance is due.'}, {'@type': 'Property', 'name': 'alertThreshold', 'schema': 'double', 'description': 'Degradation percentage that triggers a maintenance alert.', '...</code> |
  | <code>What is the best suitable digital twin interface for FatigueAnalysisEngine? Only give the interface.</code> | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Consumes stress histories from structural components and computes S‑N curve based damage using Miner’s rule. Updates accumulated damage and predicts remaining fatigue life for each critical part. Interfaces with maintenance planning to trigger alerts when thresholds are exceeded.', 'contents': [{'@type': 'Property', 'name': 'snCurveModel', 'schema': 'string'}, {'@type': 'Property', 'name': 'damageThreshold', 'schema': 'double'}, {'@type': 'Property', 'name': 'criticalComponent', 'schema': 'string'}, {'@type': 'Property', 'name': 'lastAnalysisTimestamp', 'schema': 'string'}, {'@type': 'Telemetry', 'name': 'computedDamage', 'schema': 'double'}, {'@type': 'Telemetry', 'name': 'remainingCycles', 'schema': 'double'}, {'@type': 'Telemetry', 'name': 'alertLevel', 'schema': 'string'}]}</code> | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Generates realistic wave kinematics and pressure distributions acting on the buoy. Accepts sea state parameters and outputs load spectra used by downstream fatigue components. Supports stochastic and deterministic wave generation modes.', 'contents': [{'@type': 'Property', 'name': 'significantWaveHeight', 'schema': 'double'}, {'@type': 'Property', 'name': 'peakPeriod', 'schema': 'double'}, {'@type': 'Property', 'name': 'directionality', 'schema': 'string'}, {'@type': 'Property', 'name': 'spectralModel', 'schema': 'string'}, {'@type': 'Telemetry', 'name': 'waveElevation', 'schema': {'type': 'array', 'elementSchema': 'double'}}, {'@type': 'Telemetry', 'name': 'pressureDistribution', 'schema': {'type': 'array', 'elementSchema': 'double'}}, {'@type': 'Telemetry', 'name': 'loadSpectrum', 'schema': {'type': 'array', 'elementSchema': 'double'}}]}</code>                                                 |
  | <code>What is the best suitable digital twin interface for Battery Module? Only give the interface.</code>        | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'Battery Module mirrors the power subsystem, tracking capacity, health, and real‑time charge level. It provides properties for static specifications and telemetry for dynamic state, enabling predictive maintenance and power‑aware operation. The interface also reports charging status for integration with power‑management services.', 'contents': [{'@type': 'Property', 'name': 'batteryCapacityMah', 'schema': 'integer', 'writable': False}, {'@type': 'Property', 'name': 'batteryHealthPercent', 'schema': 'double', 'writable': False}, {'@type': 'Telemetry', 'name': 'batteryLevelPercent', 'schema': 'double'}, {'@type': 'Telemetry', 'name': 'chargingStatus', 'schema': 'string'}]}</code>                                                                                                               | <code>{'@context': 'dtmi:dtdl:context;2', '@id': '', '@type': 'Interface', 'displayName': '', 'description': 'The Alert Engine component decides when to notify the wearer or external systems based on dehydration risk and device state. Configuration properties control whether alerts are active and the minimum interval between successive alerts. It emits alert events and human‑readable messages for integration with notification services.', 'contents': [{'@type': 'Property', 'name': 'alertEnabled', 'schema': 'boolean', 'writable': True}, {'@type': 'Property', 'name': 'alertCooldownMinutes', 'schema': 'integer', 'writable': True}, {'@type': 'Telemetry', 'name': 'alertTriggered', 'schema': 'boolean'}, {'@type': 'Telemetry', 'name': 'alertMessage', 'schema': 'string'}]}</code>                                                                                                                                                                                                                                            |
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
- `use_ipex`: False
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
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
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
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch  | Step  | Training Loss | Validation Loss |
|:------:|:-----:|:-------------:|:---------------:|
| 0.2020 | 500   | 2.3357        | 0.9693          |
| 0.4040 | 1000  | 0.799         | 0.5811          |
| 0.6061 | 1500  | 0.5683        | 0.4310          |
| 0.8081 | 2000  | 0.4507        | 0.3300          |
| 1.0101 | 2500  | 0.4049        | 0.2886          |
| 1.2121 | 3000  | 0.3315        | 0.2806          |
| 1.4141 | 3500  | 0.2721        | 0.2612          |
| 1.6162 | 4000  | 0.2306        | 0.2378          |
| 1.8182 | 4500  | 0.2037        | 0.2222          |
| 2.0202 | 5000  | 0.2091        | 0.2024          |
| 2.2222 | 5500  | 0.1608        | 0.2142          |
| 2.4242 | 6000  | 0.1643        | 0.2131          |
| 2.6263 | 6500  | 0.1455        | 0.1913          |
| 2.8283 | 7000  | 0.1345        | 0.1950          |
| 3.0303 | 7500  | 0.1432        | 0.1845          |
| 3.2323 | 8000  | 0.1089        | 0.1818          |
| 3.4343 | 8500  | 0.1156        | 0.1830          |
| 3.6364 | 9000  | 0.096         | 0.1720          |
| 3.8384 | 9500  | 0.0958        | 0.1641          |
| 4.0404 | 10000 | 0.0945        | 0.1670          |
| 4.2424 | 10500 | 0.0819        | 0.1663          |
| 4.4444 | 11000 | 0.0774        | 0.1691          |
| 4.6465 | 11500 | 0.0785        | 0.1617          |
| 4.8485 | 12000 | 0.0741        | 0.1600          |


### Framework Versions
- Python: 3.10.12
- Sentence Transformers: 3.3.1
- Transformers: 4.51.3
- PyTorch: 2.7.0+cu128
- Accelerate: 1.2.1
- Datasets: 3.2.0
- Tokenizers: 0.21.1

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