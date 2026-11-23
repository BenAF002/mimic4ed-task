"""
Basic datamodel for model configurations, including default feature selections 
and hyperparameter values. Not all default hyperparameters may be applicable to every model.
"""
from dataclasses import dataclass, field


DEFAULT_FEATURES = {
    "demographics": ["age", "gender"],

    "utilization": [
        "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", 
        "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d", 
    ],

    "triage": [
        "triage_temperature", "triage_heartrate", "triage_resprate", 
        "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity",
    ],
    
    "chief_complaint": [
        "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache",
        "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough", 
        "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope", 
        "chiefcom_dizziness", 
    ],

    # Charlson Comorbidity Index (CCI) features
    "cci": [
        "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia", 
        "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1", 
        "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2", 
        "cci_Cancer2", "cci_HIV", 
    ],
    
    # Elixhauser Comorbidity Index (ECI) features
    "eci": [
        "eci_Arrhythmia", "eci_Valvular", "eci_PHTN",  "eci_HTN1", "eci_HTN2", 
        "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy", 
        "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
        "eci_Anemia", "eci_Alcohol", "eci_Drugs","eci_Psychoses", "eci_Depression"
    ]
}

@dataclass
class Config:
    features: dict = field(default_factory=lambda: DEFAULT_FEATURES)

    def __post_init__(self):
        self.features_list = [f for feature_group in self.features.values() for f in feature_group]
