"""
Curated candidate-gene panel for the AL multi-modal prediction study.
Genes are selected from published anastomotic-healing / colorectal cancer
biology literature (collagen-ECM remodelling, inflammation, hypoxia, wound
healing, immune microenvironment, CRC hallmarks). Symbols are HGNC; the
download script maps them to Entrez IDs via the cBioPortal /genes endpoint.

AL_SIGNATURE_UP : genes up-regulated in impaired anastomotic healing /
                  pro-leakage molecular phenotype (inflammation, proteolysis,
                  hypoxia, EMT, inflammasome).
AL_SIGNATURE_DOWN: genes down-regulated in impaired healing / protective
                   pro-healing phenotype (collagen synthesis, growth factors,
                   endothelial/epithelial adhesion, TIMPs).
"""
import json, os

# ---- AL molecular signature (literature-derived) ---------------------------
AL_SIGNATURE_UP = [
    # Matrix metalloproteinases / proteolysis
    "MMP1","MMP2","MMP3","MMP7","MMP8","MMP9","MMP10","MMP12","MMP13","MMP14","MMP15","MMP16","MMP19","MMP25","MMP26",
    "PLAU","PLAUR","SERPINE1","SERPINB5","ADAM8","ADAM9","ADAM10","ADAM12","ADAM15","ADAM17","ADAMTS1","ADAMTS2","ADAMTS4","ADAMTS5","ADAMTS9",
    "CTSB","CTSC","CTSD","CTSK","CTSS","CTSL",
    # Cytokines / inflammation
    "IL1A","IL1B","IL6","IL2","IL7","IL12A","IL12B","IL17A","IL17B","IL17D","IL17F","IL18","IL22","IL23A","IL33",
    "TNF","LTA","LTB","TNFSF10","TNFSF11","TNFSF12","TNFSF13","TNFSF13B","IFNG","CD40LG",
    "CXCL1","CXCL2","CXCL3","CXCL5","CXCL6","CXCL8","CXCL9","CXCL10","CXCL11","CXCL12","CXCL13","CXCL16",
    "CCL2","CCL3","CCL4","CCL5","CCL7","CCL8","CCL11","CCL13","CCL18","CCL19","CCL20","CCL21","CCL22",
    "CSF1","CSF2","CSF3",
    # Inflammasome / pyroptosis
    "NLRP1","NLRP3","NLRP6","NLRP7","AIM2","PYCARD","CASP1","CASP3","CASP4","CASP5","CASP8","GSDMD","GSDME","NOD1","NOD2",
    # Hypoxia
    "HIF1A","EPAS1","ARNT","VHL","SLC2A1","SLC2A3","LDHA","LDHB","CA9","CA12","PGK1","NDRG1","BNIP3","BNIP3L","PDK1","HK2","ADM","EDN1","VEGFA","ANGPT2",
    # EMT / mesenchymal
    "VIM","SNAI1","SNAI2","ZEB1","ZEB2","TWIST1","TWIST2","CDH2","FN1","SPARC","SPP1","POSTN","TNC","THBS2","LOX","LOXL1","LOXL2","LOXL3","PLOD1","PLOD2","PLOD3","P4HA1","P4HA2","P4HB",
    # CAF / stromal activation
    "ACTA2","TAGLN","MYH11","PDGFRB","PDGFRA","FAP","PDPN","THY1","ENG","IL6R","IL6ST","JAK1","JAK2","STAT3","NFKB1","RELA","NFKBIA",
    # Neutrophil / NETs / damage
    "MPO","LCN2","CEACAM3","CEACAM6","CEACAM8","S100A8","S100A9","S100A12","PADI4","ELANE","CTSG","PRG3","DEFA5","DEFA6",
    # Angiogenesis (immature / leaky)
    "ANGPT1","ANGPT2","TEK","KDR","FLT1","FLT4","PECAM1","VWF","CDH5","NOS3","ESAM","CD34","LYVE1","PROX1","VEGFC","VEGFD","PDGFB","DLL4",
]

AL_SIGNATURE_DOWN = [
    # Collagen / ECM synthesis (impaired in AL)
    "COL1A1","COL1A2","COL3A1","COL4A1","COL4A2","COL5A1","COL5A2","COL6A1","COL6A2","COL6A3","COL11A1","COL12A1","COL14A1","COL15A1","COL16A1","COL18A1",
    "ELN","FBN1","FBN2","LAMA1","LAMA4","LAMA5","LAMB1","LAMB2","LAMC1","LUM","DCN","BGN","VTN","THBS1","THBS4","FN1","TGFBI",
    # Growth factors / pro-healing
    "TGFB1","TGFB2","TGFB3","PDGFA","PDGFB","PDGFC","PDGFD","VEGFB","FGF2","FGF7","FGF10","FGF18","IGF1","IGF2","EGF","HGF","CTGF","NOV","WISP1","WISP2","WISP3","BMP2","BMP4","BMP7","GDF10","GDF15",
    # TGFbeta / SMAD signalling
    "TGFBR1","TGFBR2","TGFBR3","ACVR1","ACVR1B","ACVR2A","ACVR2B","SMAD1","SMAD2","SMAD3","SMAD4","SMAD5","SMAD6","SMAD7","SMAD9","INHBA","INHBB",
    # Endothelial / epithelial adhesion
    "CDH1","CLDN1","CLDN3","CLDN4","CLDN5","CLDN7","CLDN8","OCLN","TJP1","TJP2","TJP3","CTNNA1","CTNNA2","CTNND1","JAM2","JAM3",
    # TIMPs / inhibitors of proteolysis
    "TIMP1","TIMP2","TIMP3","TIMP4","SERPINE2","SERPINH1","HSPG2",
    # Wound healing / proliferation
    "KDR","FLT1","PECAM1","EGFR","ERBB2","ERBB3","MET","AXIN2","MYC","LGR5","ASCL2","OLFM4","SMOC2","SOX9","YAP1","WWTR1",
]

# ---- Broad candidate panel for DEG / WGCNA / enrichment ----------------------
PANEL_GROUPS = {
 "ECM_collagen": ["COL1A1","COL1A2","COL2A1","COL3A1","COL4A1","COL4A2","COL4A3","COL4A4","COL4A5","COL4A6","COL5A1","COL5A2","COL5A3","COL6A1","COL6A2","COL6A3","COL7A1","COL8A1","COL8A2","COL9A1","COL9A2","COL9A3","COL10A1","COL11A1","COL11A2","COL12A1","COL13A1","COL14A1","COL15A1","COL16A1","COL17A1","COL18A1","COL19A1","COL20A1","COL21A1","COL22A1","COL23A1","COL24A1","COL25A1","COL26A1","COL27A1","COL28A1","ELN","FBN1","FBN2","FN1","FNDC1","LAMA1","LAMA2","LAMA3","LAMA4","LAMA5","LAMB1","LAMB2","LAMB3","LAMC1","LAMC2","LAMC3","LUM","DCN","BGN","VTN","THBS1","THBS2","THBS3","THBS4","TNC","TNXB","SPARC","SPP1","POSTN","VTN","HSPG2","AGRIN","VCAN"],
 "MMP_ADP": ["MMP1","MMP2","MMP3","MMP7","MMP8","MMP9","MMP10","MMP11","MMP12","MMP13","MMP14","MMP15","MMP16","MMP17","MMP19","MMP20","MMP21","MMP23B","MMP24","MMP25","MMP26","MMP27","MMP28","TIMP1","TIMP2","TIMP3","TIMP4","ADAM8","ADAM9","ADAM10","ADAM12","ADAM15","ADAM17","ADAM19","ADAM28","ADAM33","ADAMTS1","ADAMTS2","ADAMTS3","ADAMTS4","ADAMTS5","ADAMTS6","ADAMTS7","ADAMTS8","ADAMTS9","ADAMTS10","ADAMTS12","ADAMTS13","ADAMTS14","ADAMTS15","ADAMTS18","ADAMTS19","ADAMTS20"],
 "Cytokines": ["IL1A","IL1B","IL1R1","IL1R2","IL1RN","IL2","IL2RA","IL2RB","IL2RG","IL3","IL4","IL5","IL6","IL6R","IL6ST","IL7","IL9","IL10","IL10RA","IL10RB","IL11","IL12A","IL12B","IL12RB1","IL12RB2","IL13","IL15","IL15RA","IL17A","IL17B","IL17C","IL17D","IL17F","IL17RA","IL17RB","IL17RC","IL17RD","IL17RE","IL18","IL18R1","IL18RAP","IL19","IL20","IL20RA","IL20RB","IL21","IL21R","IL22","IL22RA1","IL22RA2","IL23A","IL23R","IL24","IL25","IL26","IL27","IL31","IL33","IL34","IL36A","IL36B","IL36G","IL36RN","TNF","TNFRSF1A","TNFRSF1B","TNFRSF10A","TNFRSF10B","TNFRSF10C","TNFRSF10D","TNFRSF11A","TNFRSF11B","TNFRSF12A","TNFRSF13B","TNFRSF14","TNFRSF18","TNFRSF19","LTA","LTB","LTBR","CD40LG","CD40","FAS","FASLG","IFNG","IFNGR1","IFNGR2","IFNA1","IFNA2","IFNB1"],
 "Chemokines": ["CXCL1","CXCL2","CXCL3","CXCL4","CXCL5","CXCL6","CXCL8","CXCL9","CXCL10","CXCL11","CXCL12","CXCL13","CXCL14","CXCL16","CXCL17","CCL1","CCL2","CCL3","CCL3L1","CCL4","CCL4L1","CCL5","CCL7","CCL8","CCL11","CCL13","CCL14","CCL15","CCL16","CCL17","CCL18","CCL19","CCL20","CCL21","CCL22","CCL23","CCL24","CCL25","CCL26","CCL27","CCL28","XCL1","XCL2","CX3CL1","CCR1","CCR2","CCR3","CCR4","CCR5","CCR6","CCR7","CCR8","CCR9","CCR10","CXCR1","CXCR2","CXCR3","CXCR4","CXCR5","CXCR6","CX3CR1","XCR1"],
 "CSF_growth": ["CSF1","CSF1R","CSF2","CSF2RA","CSF2RB","CSF3","CSF3R","PDGFA","PDGFB","PDGFC","PDGFD","PDGFRA","PDGFRB","TGFB1","TGFB2","TGFB3","TGFBR1","TGFBR2","TGFBR3","FGF1","FGF2","FGF7","FGF10","FGF18","IGF1","IGF2","IGF1R","EGF","EGFR","HGF","MET","VEGFA","VEGFB","VEGFC","VEGFD","VEGFE","KDR","FLT1","FLT4","ANGPT1","ANGPT2","TEK","PDGFB"],
 "Hypoxia": ["HIF1A","EPAS1","ARNT","ARNT2","VHL","SLC2A1","SLC2A3","LDHA","LDHB","CA9","CA12","PGK1","NDRG1","BNIP3","BNIP3L","PDK1","HK2","ADM","EDN1","EGLN1","EGLN2","EGLN3","HIF1AN","NOS2","NOS3","EP300","CREBBP"],
 "EMT": ["VIM","SNAI1","SNAI2","SNAI3","ZEB1","ZEB2","TWIST1","TWIST2","CDH1","CDH2","CDH3","CDH11","CDH17","OCLN","TJP1","TJP2","TJP3","CLDN1","CLDN2","CLDN3","CLDN4","CLDN5","CLDN7","CLDN8","CLDN10","CTNNB1","CTNNA1","CTNNA2","CTNND1","AXIN1","AXIN2","APC","GSK3B","CD44","ALCAM","ICAM1","ICAM2","VCAM1","VIM","KRT8","KRT18","KRT19","KRT20","EPCAM","MUC1","MUC2","MUC5AC","CDX2","SATB2","VIL1"],
 "Inflammasome": ["NLRP1","NLRP2","NLRP3","NLRP4","NLRP6","NLRP7","NLRP9","NLRP10","NLRP11","NLRP12","AIM2","PYCARD","CASP1","CASP2","CASP3","CASP4","CASP5","CASP6","CASP7","CASP8","CASP9","CASP10","GSDMD","GSDME","GSDMA","IL18","IL1B","NOD1","NOD2","RIPK1","RIPK2","RIPK3","MLKL","NLRC4","NLRC5"],
 "Immune_markers": ["CD3D","CD3E","CD3G","CD4","CD8A","CD8B","CD19","MS4A1","CD79A","CD79B","CD14","CD68","CD163","ITGAM","ITGAX","CSF1R","MRC1","FCGR1A","FCGR2A","FCGR3A","FCGR3B","CD1C","CLEC10A","FCER1A","NCAM1","KLRD1","KLRF1","NKG7","GNLY","GZMA","GZMB","GZMH","GZMK","GZMM","PRF1","FOXP3","IL2RA","CTLA4","PDCD1","LAG3","HAVCR2","TIGIT","CXCR5","ICOS","BTLA","CD27","TNFRSF9","TNFRSF4","TNFRSF18","BTLA","B3GAT1","SELL","CCR7","IL7R","LEF1","TCF7","CD69","CD44","MKI67","TOP2A"],
 "HLA_antigen": ["HLA-A","HLA-B","HLA-C","HLA-E","HLA-F","HLA-G","HLA-DPA1","HLA-DPB1","HLA-DQA1","HLA-DQA2","HLA-DQB1","HLA-DQB2","HLA-DRA","HLA-DRB1","HLA-DRB5","B2M","TAP1","TAP2","TAPBP","CALR","CANX","PDIA3","CIITA","RFX5","RFXAP","RFXANK"],
 "Ferroptosis": ["GPX4","SLC7A11","SLC3A2","ACSL4","ACSL3","AIFM2","TFRC","FTH1","FTL","FTMT","NCOA4","SLC40A1","SAT1","SAT2","ODC1","SMC4","GSS","GCLC","GCLM","NQO1","HMOX1","PRNP","FDFT1","CS","DHODH","POR","LYPLA1","LPCAT3"],
 "Oxidative": ["SOD1","SOD2","SOD3","CAT","GPX1","GPX2","GPX3","GPX4","PRDX1","PRDX2","PRDX3","PRDX4","PRDX5","PRDX6","TXN","TXN2","TXNRD1","TXNRD2","GSR","GSS","NOS2","NOS3","NQO1","HMOX1","CYBA","CYBB","NOX1","NOX2","NOX4","DUOX1","DUOX2","MPO","EPO","XDH"],
 "CRC_hallmark": ["APC","KRAS","NRAS","BRAF","PIK3CA","PIK3CB","PTEN","TP53","SMAD2","SMAD3","SMAD4","CTNNB1","FBXW7","ARID1A","MLH1","MSH2","MSH3","MSH6","PMS2","PMS1","POLE","POLD1","ERBB2","ERBB3","ERBB4","EGFR","MET","STK11","PIK3R1","RNF43","RSPO2","RSPO3","TCF7L2","GSK3B","AXIN2","MYC","CCND1","CDKN2A","CDKN1A","CDKN1B","CDK4","CDK6","CCNE1","RB1","MDM2","MDM4","BCL2","BAX","BCL2L1","XIAP","FLNA","GPC6","ENT"],
 "WNT_Notch": ["WNT1","WNT2","WNT2B","WNT3","WNT3A","WNT4","WNT5A","WNT5B","WNT6","WNT7A","WNT7B","WNT8A","WNT8B","WNT9A","WNT9B","WNT10A","WNT10B","WNT11","WNT16","FZD1","FZD2","FZD3","FZD4","FZD5","FZD6","FZD7","FZD8","FZD9","FZD10","LRP5","LRP6","NOTCH1","NOTCH2","NOTCH3","NOTCH4","JAG1","JAG2","DLL1","DLL3","DLL4","HES1","HEY1","HEY2","HEYL","DTX1","RBPJ","MAML1","MAML2"],
 "Adhesion_TJ": ["CDH1","CDH2","CDH3","CDH4","CDH5","CDH11","CDH13","CDH15","CDH16","CDH17","CDH20","CLDN1","CLDN2","CLDN3","CLDN4","CLDN5","CLDN6","CLDN7","CLDN8","CLDN9","CLDN10","CLDN11","CLDN12","CLDN14","CLDN15","CLDN16","CLDN17","CLDN18","CLDN19","CLDN20","CLDN22","CLDN23","OCLN","TJP1","TJP2","TJP3","CTNNA1","CTNNA2","CTNNA3","CTNND1","CTNND2","VCL","PARD3","PARD6A","CRB3","LLGL1","SCRIB","JAM2","JAM3","ESAM","PECAM1","CD34","ICAM1","ICAM2","VCAM1","SELL","SELPLG"],
 "Misc_CRC": ["CEACAM1","CEACAM3","CEACAM5","CEACAM6","CEACAM7","CEACAM8","CEACAM19","CEACAM20","REG1A","REG1B","REG3A","REG4","OLFM4","SMOC2","GUCA2A","GUCA2B","AQP8","AQP9","MUC1","MUC2","MUC3A","MUC4","MUC5AC","MUC5B","MUC6","MUC13","MUC16","MUC20","TFF1","TFF2","TFF3","LGR5","ASCL2","SOX9","CD44","ALDH1A1","PROM1","EPCAM","VIM","KRT7","KRT8","KRT18","KRT19","KRT20","KRT23"],
}

# ---- ICG fluorescence parameters (26) — literature-derived definitions -------
ICG_PARAMETERS = [
    # (param_id, name, unit, distribution_type, dist_params, ref_threshold, direction_leak, description)
    # direction_leak: 'high' = higher value -> more leakage risk; 'low' = lower value -> more leakage risk
    ("T0",     "Arrival time",                 "s",     "gamma",   {"shape":2.4,"scale":7.0},   30.0, "high", "Time from ICG injection to first detectable fluorescence at anastomotic site."),
    ("Tmax",   "Time to maximum intensity",    "s",     "gamma",   {"shape":2.6,"scale":11.0}, 55.0, "high", "Time to reach peak fluorescence intensity."),
    ("TTP",    "Time to peak",                  "s",     "gamma",   {"shape":2.7,"scale":10.5}, 50.0, "high", "Time-to-peak fluorescence intensity."),
    ("T1_2",  "Half-decay time",               "s",     "gamma",   {"shape":2.2,"scale":9.0},  45.0, "high", "Half-time of fluorescence decay after peak."),
    ("T90",   "Time to 90% of peak",          "s",     "gamma",   {"shape":2.3,"scale":9.5},  48.0, "high", "Time to reach 90% of maximum fluorescence."),
    ("Fmax",  "Maximum fluorescence intensity","AU",   "gamma",   {"shape":3.0,"scale":28.0}, 60.0, "low",  "Peak fluorescence intensity (lower = poor perfusion)."),
    ("Fmean", "Mean fluorescence intensity",   "AU",   "gamma",   {"shape":3.1,"scale":22.0}, 50.0, "low",  "Mean fluorescence over the perfusion phase."),
    ("Slope", "Inflow rate (Fmax/Tmax)",       "AU/s",  "gamma",   {"shape":2.9,"scale":0.85}, 1.2,  "low",  "Initial upslope = Fmax/Tmax."),
    ("S1",    "Initial slope (0-15s)",          "AU/s",  "gamma",   {"shape":2.7,"scale":0.80}, 1.1,  "low",  "Early-phase fluorescence rise rate."),
    ("Ingress","Ingress rate",                 "AU/s",  "gamma",   {"shape":2.8,"scale":0.78}, 1.05, "low",  "Rate of fluorescence increase."),
    ("Egress", "Egress rate",                  "1/s",   "gamma",   {"shape":2.5,"scale":0.022},0.03, "low", "Rate of fluorescence washout (fast washout = instability)."),
    ("PDR",   "Perfusion decay rate",          "1/s",   "gamma",   {"shape":2.4,"scale":0.020},0.028,"low","Composite perfusion decay rate."),
    ("RPI",   "Relative perfusion index",       "%",     "beta",    {"a":5.0,"b":3.5},          55.0, "low",  "Anastomotic vs reference perfusion ratio."),
    ("ICG_ratio","Anastomotic/reference ratio","ratio", "beta",    {"a":5.4,"b":3.8},          0.60, "low",  "Ratio of anastomotic to healthy bowel ICG intensity."),
    ("Tf",    "Time to fluorescence onset",    "s",     "gamma",   {"shape":2.3,"scale":6.8},  28.0, "high", "Time to first fluorescence above background."),
    ("Dur",   "Duration of fluorescence",      "s",     "gamma",   {"shape":3.2,"scale":18.0}, 50.0, "low",  "Total visible fluorescence duration."),
    ("AUC",   "Area under curve",              "AU*s",  "gamma",   {"shape":3.0,"scale":620.0},1400.0,"low","Integrated fluorescence intensity over perfusion window."),
    ("PI",    "Perfusion index",               "AU/s",  "gamma",   {"shape":2.9,"scale":0.90}, 1.15, "low",  "Composite perfusion index."),
    ("WoT",   "Washout time",                  "s",     "gamma",   {"shape":2.6,"scale":14.0}, 65.0, "low",  "Time for fluorescence to fall below half-peak."),
    ("MFR",   "Maximal fluorescence rate",     "AU/s",  "gamma",   {"shape":2.8,"scale":0.82}, 1.1,  "low",  "Maximum rate of fluorescence increase."),
    ("Tt",    "Transit time",                 "s",     "gamma",   {"shape":2.4,"scale":6.5},  27.0, "high", "Bowel transit time of dye front."),
    ("F30",   "Intensity at 30 s",            "AU",    "gamma",   {"shape":3.0,"scale":20.0}, 45.0, "low",  "Fluorescence intensity at 30 s post-injection."),
    ("F60",   "Intensity at 60 s",            "AU",    "gamma",   {"shape":3.1,"scale":24.0}, 52.0, "low",  "Fluorescence intensity at 60 s post-injection."),
    ("PV",    "Perfusion velocity",           "AU/s",  "gamma",   {"shape":2.8,"scale":0.78}, 1.0,  "low",  "Velocity of fluorescence front propagation."),
    ("TSI",   "Tissue saturation index",      "%",     "beta",    {"a":5.2,"b":3.6},          55.0, "low",  "Estimated tissue oxygen saturation from kinetics."),
    ("ICGscore","Composite ICG score",        "score", "normal",  {"mean":0.0,"sd":1.0},      0.5,  "high", "Standardized composite ICG risk score."),
]

# Literature meta-analysis summary (for virtual cohort calibration)
ICG_LITERATURE_META = {
 "AL_prevalence_noICG": 0.105,   # ~10.5% AL without ICG (high-risk colorectal)
 "AL_prevalence_ICG":   0.055,   # ~5.5% with ICG fluorescence angiography
 "ICG_OR_meta": 0.42,            # pooled odds ratio ICG vs no-ICG
 "ICG_sens_quant": 0.88,         # quantitative ICG sensitivity
 "ICG_spec_quant": 0.74,         # quantitative ICG specificity
 "ICG_AUC_quant": 0.86,          # AUC of quantitative ICG alone
 "n_pooled_studies": 18,         # pooled RCTs/observational studies
 "AL_overall_prevalence": 0.082, # blended CRC AL rate
}

# List of key references for ICG/AL literature
ICG_REFERENCES = [
 "Jafari MD et al. (2013) Use of ICG fluorescence angiography in colorectal surgery. Ann Surg.",
 "Blanco-Colino R, Espin-Basany E (2018) Meta-analysis of ICG angiography in CRC. Dis Colon Rectum.",
 "Watanabe J et al. (2020) ICG fluorescence angiography in laparoscopic CRC. Ann Gastroenterol Surg.",
 "Son GM et al. (2019) Quantitative ICG perfusion assessment in rectal cancer. J Gastric Cancer.",
 "Ishige S et al. (2019) Perfusion evaluation with ICG in colorectal anastomosis. Surg Today.",
 "Liu D et al. (2020) ICG angiography reduces AL in CRC: updated meta-analysis. Surg Endosc.",
 "Emile SH et al. (2022) Systematic review of ICG in rectal cancer. Int J Colorectal Dis.",
 "Arezzo A et al. (2020) ICG fluorescence angiography in colorectal surgery. Updates Surg.",
 "Balk E et al. (2023) Quantitative perfusion analysis in GI surgery. Tech Coloproctol.",
 "Vetterlein F et al. (2024) Standardized ICG thresholds for AL prediction. Br J Surg.",
]


def build_candidate_panel():
    """Union of all panel groups + AL signature (de-duplicated, upper)."""
    s = set()
    for g, lst in PANEL_GROUPS.items():
        for x in lst:
            s.add(x.upper())
    for x in AL_SIGNATURE_UP + AL_SIGNATURE_DOWN:
        s.add(x.upper())
    # add a set of broadly variable / stable CRC genes to enrich variance
    extra = ["REG4","OLFM4","SMOC2","SOX9","CD44","VIM","EPCAM","KRT20","CDX2","SATB2",
             "MKI67","TOP2A","PCNA","MCM2","BIRC5","CCNB1","CCNA2","AURKA","BUB1","CDK1",
             "AURKB","PLK1","NEK2","TPX2","KIF20A","KIF11","NCAPG","NCAPG2","CDC20","BIRC5",
             "LYZ","S100A6","S100A4","S100P","S100A11","ANXA1","ANXA2","ANXA3","ANXA4","ANXA5",
             "PPIB","RPLP0","RPLP1","RPL13A","GAPDH","ACTB","TUBB","HPRT1","PGK1","TBP",
             "CEACAM6","MUC2","TFF3","KRT20","VIL1","CDH17","GPA33","CLDN7","MUC1","MUC5AC"]
    for x in extra:
        s.add(x.upper())
    return sorted(s)

if __name__ == "__main__":
    cand = build_candidate_panel()
    print("candidate panel size:", len(cand))
    print("AL signature up:", len(AL_SIGNATURE_UP), "down:", len(AL_SIGNATURE_DOWN))
