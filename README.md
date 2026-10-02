

# SANKET — Compressor Anomaly Detection and Operator Decision Support

**Live demo:** [sanket-anamoly-detector-operator-decision-supporter.streamlit.app](https://sanket-anamoly-detector-operator-decision-supporter.streamlit.app/)

**Dataset:** [Refinery Compressor Sensor Data (RCSD-1YD) — Zenodo](https://zenodo.org/records/14866092)


An unsupervised anomaly detection system for a refinery centrifugal compressor, built with real production-plant data. The system learns what normal operation looks like, flags periods when the compressor behaved abnormally, and ranks the sensors that contributed most to each anomaly — providing root-cause support, not root-cause proof.

---

## 1. Problem Statement

A refinery centrifugal compressor runs continuously and is monitored by 25 sensors (axial displacement, vibration, temperatures, pressures, and speed). Operators cannot watch all 25 signals at once, and subtle drift — such as thrust-bearing wear before a shutdown — can be missed until damage has already occurred.

This project builds a monitoring system that:

1. Learns normal steady-state behaviour from historical data.
2. Flags periods when the compressor deviates from that behaviour.
3. Ranks the sensors that changed most during each anomaly.
4. Provides an operator-facing dashboard to investigate flagged periods.

The system is designed to support engineers, not replace them. It identifies the pattern of deviations; the engineer interprets the pattern using chemical engineering knowledge.

---

## 2. Industrial Background

Centrifugal compressors are critical rotating equipment in refineries. In hydrogen service, they supply the pressure and flow required by hydrotreating and hydrocracking units. The compressor monitored here is a BCL 509/A type driven by a steam turbine, running continuously at approximately 6,200–6,900 RPM.

Key failure modes of centrifugal compressors:

- Thrust bearing wear — axial displacement rises, vibration rises, bearing temperature rises.
- Lube oil degradation — oil temperature rises, oil pressure drops, bearing temperatures rise.
- Impeller fouling — axial displacement rises moderately, vibration rises, but temperatures and pressures remain near normal.
- Seal oil leakage — seal oil pressure drops, with little change elsewhere.
- Filter clogging — differential pressure rises slowly.

Detecting these patterns early allows planned maintenance instead of emergency shutdowns.

---

## 3. Dataset Source

Name: Refinery Compressor Sensor Data, One-Year Dataset (RCSD-1YD)

Provenance: Motor Oil Hellas, Corinth Refineries, Greece. Data collected from the plant DCS throughout 2022.

Classification: A — Real production-plant operational data. This is not simulated and not from a test facility.

Link: https://zenodo.org/records/14866092

Citation: Ntafalias, A., Tsakanikas, S., Papadopoulos, P., et al. (2025). "Descriptor: Refinery Compressor Sensor Data, One-Year Dataset (RCSD-1YD)." IEEE Data Descriptions.

Properties:

- Timestamps: 35,036
- Sensors: 25
- Sampling interval: 15 minutes
- Time span: 1 January 2022 to 31 December 2022
- Missing values: None
- File size: 11.1 MB (Excel)

Sensor tags (ISA-5.1 convention):

- ZI — axial displacement (4 channels)
- PI — pressure (4 channels)
- PDI — differential pressure (1 channel)
- TI — temperature (11 channels)
- XI — vibration (4 channels)
- SI — speed (1 channel)

---

## 4. Dataset Limitations

The dataset documentation does not include the P&ID (Piping and Instrumentation Diagram). This means:

- The exact physical connection of each temperature and pressure tag is not confirmed. For example, whether temp_7 is a bearing temperature or a lube oil temperature cannot be stated with certainty.
- Sensor units are inferred from typical compressor instrumentation, not confirmed from the documentation.

The tag type (ZI, PI, TI, XI, SI) is known with confidence. The specific process connection of each channel is inferred from typical centrifugal compressor layouts and observed value ranges.

The compressor ran in a stable configuration during the year, with two planned shutdowns (June and September). The dataset therefore contains no labelled fault events. Anomaly detection is entirely unsupervised.

---

## 5. Chemical Engineering Principle

A centrifugal compressor converts mechanical energy (from a steam turbine) into gas pressure. The rotor spins at high speed and is supported by:

- Radial bearings — carry the rotor's weight.
- Thrust bearings — prevent the rotor from moving along its axis.
- Lube oil system — provides a hydrodynamic film between the shaft and the bearings, and removes frictional heat.
- Seal oil system — prevents gas from leaking along the shaft in hydrogen service.

Frictional heat is continuously generated at the bearings and carried away by the lube oil. Any change in bearing condition (wear, misalignment, or film breakdown) changes the balance of heat generation and removal, which propagates to:

- Axial displacement (shaft moving along its axis)
- Vibration (shaft moving radially)
- Bearing temperatures (friction)
- Oil pressures (film thickness)

By monitoring all these variables together, it is possible to identify which subsystem is degrading.

---

## 6. Methodology

Real industrial data (35,036 timestamps × 25 sensors)
        ↓
Data cleaning (timestamp conversion, column renaming)
        ↓
Steady-state identification (speed between 5500 and 7000 RPM, with minimum block length 6 hours)
        ↓
Train Isolation Forest on steady-state operation only
        ↓
Apply model to the whole year
        ↓
Compute anomaly score for every timestamp
        ↓
Persistence filter (keep only periods lasting >= 1 hour)
        ↓
Compute group-level and per-sensor z-scores per period
        ↓
Rank contributors and interpret with chemical engineering
        ↓
Operator dashboard (Streamlit)

---

## 7. Data Preprocessing

1. Convert timestamp from text to datetime for time-based operations.
2. Rename columns from 75ZI800BA.pv to axial_1 etc., for readability.
3. Define steady-state: rows where speed is between 5500 and 7000 RPM.
4. Apply block filter: only keep steady-state runs of at least 6 hours. This removes brief restart transients.
5. Split: training set = steady-state rows only (34,014 rows, 97% of the year). Test set = the full year.

No scaling is applied. Isolation Forest does not require it because it splits one feature at a time.

---

## 8. ML Model

Algorithm: Isolation Forest (unsupervised).

Why Isolation Forest:

- Unsupervised — no fault labels are required.
- Designed for anomaly detection — it isolates rare and different points.
- Handles multivariate data naturally.
- Interpretable — per-variable z-scores can be computed for each anomaly.
- Fast — trains in seconds on 34,000 rows.

Why not Random Forest: Random Forest is a supervised algorithm requiring labelled training data. In a real plant, labelled fault data are rare or unavailable. Random Forest is used in the author's S-Zorb project, and this project deliberately uses a different methodology.

Why not a neural network: A neural network would require much more data, much longer training, and would be a black box. Isolation Forest is simpler, faster, and easier to explain to an operator.

Parameters:

- n_estimators = 200
- contamination = 0.01 (expected fraction of anomalies)
- random_state = 42 (reproducibility)

What the model learns: The joint distribution of all 25 sensors during steady-state normal operation. It does not learn any specific fault signature.

Anomaly score: The average number of random splits required to isolate a point across all trees. Points isolated quickly (short path) get high scores. Scores above a threshold are flagged as anomalies.

---

## 9. Results

30 anomaly periods lasting at least 1 hour were detected across the year. They group into four categories:

- Shutdowns (June 18–22, Aug 30–Sep 2) — Temperature, pressure, speed all drop sharply.
- Restarts (June 22–23, Sep 2–4) — Speed overshoots, temperatures warm back up.
- Thermal events (Apr 27–28, May 11–12) — Temperatures rise together, nothing else changes.
- Mechanical/operational events (Jun 17 axial, Apr 28 pressure) — Single group dominates.

The most important period: 17 June 2022, approximately 24 hours before the June shutdown. Axial displacement channel axial_4 reached +8 standard deviations above its normal value. This is a classic pre-failure signature of thrust bearing wear. In a real plant, this alert would have triggered a bearing inspection, allowing planned maintenance instead of an emergency shutdown.

The second most important period: 28 April 2022. pressure_1 dropped -36 standard deviations below normal for approximately 1 hour. This is not a gradual drift — it is a discrete event, consistent with a valve stroke, filter change, or brief process upset.

See results/plots/group_heatmap.png for the summary figure.
---

## 10. Limitations of Sanket

- The dataset has no labelled faults. Precision, recall, and F1 cannot be computed. Evaluation is qualitative and physical.
- The compressor ran stably during the year. It is not a dataset with dramatic failures; the anomalies are subtle.
- Sensor identities are partly inferred. The P&ID was not released, so the exact physical meaning of each temperature and pressure tag is not confirmed.
- No causal claims. The system identifies leading contributors, not causes.
- A real plant would layer additional filtering. Persistence filters, severity thresholds, and maintenance-state awareness would reduce alert volume further.
- Model retraining required over time. As the machine ages, the definition of "normal" shifts. The model should be retrained periodically on recent steady-state data.

---
## 11. Business Model

**Customer:** Refineries and petrochemical plants running critical rotating equipment.

**Value:** SANKET flags degradation hours to days before failure — turning unplanned shutdowns into scheduled work.

**Revenue:** Annual subscription per machine ($30k–$80k), plus one-time integration ($50k–$150k). Enterprise tier ($250k–$500k per site per year).

**Channel:** Sold through Honeywell UOP's Connected Performance Services portfolio.
A subscription that watches refinery compressors continuously and tells operators what to investigate before failure.

---
## 15. How to Run

1. Clone the repository.
2. Install dependencies: pip install -r requirements.txt
3. Download the raw dataset from https://zenodo.org/records/14866092
   Save as: data/raw/Centrifugal compressor _2022.xlsx
4. Run the pipeline:
   python src/clean_data.py
   python src/prepare_data.py
   python src/train_model.py
   python src/plot_anomalies.py
   python src/find_contributors.py
   python src/filter_and_group.py
   python src/make_plots.py
5. Launch the dashboard:
   python -m streamlit run app/dashboard.py

