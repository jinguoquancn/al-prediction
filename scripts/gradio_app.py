"""
gradio_app.py — Interactive risk calculator for anastomotic leakage.
Run:  python gradio_app.py   ->  http://127.0.0.1:7860
Inputs: molecular AL signature score, clinical risk index, 3 core ICG parameters.
Predicts AL probability with the fused model (logistic nomogram weights).
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg

NM = json.load(open(os.path.join(cfg.RESULTS, "nomogram_model.json"))) if \
     os.path.exists(os.path.join(cfg.RESULTS, "nomogram_model.json")) else \
     {"intercept": -2.9, "coefficients": {"mol_z": 0.9, "clin_z": 0.6, "icg_z": 1.1}}

def predict(mol_score, age, stage, t_max, slope, p10_ratio):
    """Compute AL probability from raw inputs (z-scores derived internally)."""
    mol_z = (mol_score - 0.0) / 1.0  # AL signature score approx z
    stage_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
    st = stage_map.get(stage, 2) / 4.0
    age_r = np.clip((age - 50) / 20, -1, 2)
    clin_r = 0.5 * st + 0.5 * age_r
    clin_z = (clin_r - 0.9) / 0.5
    # composite ICG z (higher = leak direction: longer times, lower intensity)
    t_z = (t_max - 26.0) / 8.0
    s_z = (slope - 0.55) / 0.2
    p_z = (1.1 - p10_ratio) / 0.45
    icg_z = (t_z + s_z + p_z) / 3.0
    coefs = NM["coefficients"]
    lp = NM["intercept"] + coefs.get("mol_z", 0.9) * mol_z + \
         coefs.get("clin_z", 0.6) * clin_z + coefs.get("icg_z", 1.1) * icg_z
    p = 1 / (1 + np.exp(-lp))
    risk = "HIGH" if p >= 0.15 else ("INTERMEDIATE" if p >= 0.08 else "LOW")
    return float(p), risk, f"molecular z={mol_z:.2f}, clinical z={clin_z:.2f}, ICG z={icg_z:.2f}"

def main():
    try:
        import gradio as gr
    except ImportError:
        print("gradio not installed; run: pip install gradio")
        return
    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="Molecular AL signature score (z)"),
            gr.Slider(18, 95, value=62, step=1, label="Age (years)"),
            gr.Radio(["I", "II", "III", "IV"], value="II", label="AJCC stage"),
            gr.Slider(5, 60, value=26.0, step=0.5, label="ICG T_max (time to peak, s)"),
            gr.Slider(0.1, 1.2, value=0.55, step=0.01, label="ICG slope (inflow rate, AU/s)"),
            gr.Slider(0.2, 2.5, value=1.1, step=0.01, label="ICG P10/P90 perfusion index"),
        ],
        outputs=[
            gr.Label(num_top_classes=None, label="Predicted AL probability"),
            gr.Textbox(label="Risk category (thresholds: 8% / 15%)"),
            gr.Textbox(label="Component contributions"),
        ],
        title="Anastomotic Leakage Multimodal Risk Calculator",
        description=("Fusion of molecular AL-risk signature, clinical risk index, and "
                      "standardized ICG fluorescence parameters. Virtual-cohort "
                      "calibrated model — for research use only."),
    )
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)

if __name__ == "__main__":
    main()
