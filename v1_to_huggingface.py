import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import numpy as np
import os
import random
import tempfile
from huggingface_hub import HfApi, hf_hub_download

# --- CONFIGURATION DU STOCKAGE ---
REPO_ID = "LLMGAB/Fichier_notes" 
DATA_FILENAME = "NOTES_CP.csv"
DATA_PATH = Path(DATA_FILENAME)
HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi()

def load_data():
    cols = ["eleve", "date", "matiere", "devoir", "note", "commentaire"]
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=DATA_FILENAME, repo_type="dataset", token=HF_TOKEN)
        df = pd.read_csv(path)
        return df[cols]
    except:
        if DATA_PATH.exists(): return pd.read_csv(DATA_PATH)
        return pd.DataFrame(columns=cols)

def save_data(df_to_save):
    df_to_save.to_csv(DATA_PATH, index=False)
    if HF_TOKEN:
        try:
            api.upload_file(path_or_fileobj=str(DATA_PATH), path_in_repo=DATA_FILENAME,
                            repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
        except: pass

df = load_data()

# --- GÉNÉRATEUR DE COMMENTAIRES AVEC ANALYSE DE PROGRESSION (DELTA) ---

#def get_smart_comment(eleve, note, matiere, genre):
#    global df
#    if note == "Absent" or not eleve or not matiere: 
#        return "Absent(e) lors de l'évaluation."    
#    try:
#        n_actuelle = float(note)
#    except:
#        return ""

#    # --- CALCUL DU DELTA ---
#    delta = None
    # On cherche les notes passées de cet élève dans cette matière
#    prec_df = df[(df["eleve"] == eleve) & (df["matiere"] == matiere) & (df["note"] != "Absent")].copy()
    
#    if not prec_df.empty:
#        prec_df["date"] = pd.to_datetime(prec_df["date"])
#        prec_df = prec_df.sort_values("date")
        # On récupère la dernière note enregistrée
#        derniere_note = float(prec_df.iloc[-1]["note"])
#        delta = n_actuelle - derniere_note

    # --- ACCORDS ---
#    e = "e" if genre == "F" else ""
#    il_elle = "Elle" if genre == "F" else "Il" if genre == "M" else "L'élève"
    
    # --- BANQUE DE PHRASES DYNAMIQUE ---
#    comm_final = ""

    # 1. Analyse de la dynamique (Delta)
#    if delta is not None:
#        if delta > 0:
#            comm_final = random.choice([
#                f"Quelle belle progression ! (+{delta:g} pts). ",
#                f"Bravo, les efforts portent leurs fruits avec une hausse de {delta:g} points ! ",
#                f"Une dynamique très positive par rapport au dernier travail. "
#            ])
#        elif delta < 0:
#            comm_final = random.choice([
#                f"Une petite baisse ce coup-ci (-{abs(delta):g} pts), mais on reste mobilisé{e}. ",
#                f"Ce résultat est en retrait, ne te décourage pas. ",
#                f"Attention au relâchement, {il_elle.lower()} peut mieux faire avec plus de rigueur. "
#            ])
#        else:
#            comm_final = "Résultat très stable. "
#    else:
#        comm_final = "Premier bilan dans cette matière. "

    # 2. Analyse du niveau absolu
#    if n_actuelle >= 18:
#        comm_final += "Travail d'une qualité exceptionnelle."
#    elif n_actuelle >= 14:
#        comm_final += f"C'est un très bon résultat, {il_elle.lower()} est sur la bonne voie."
#    elif n_actuelle >= 10:
#        comm_final += "L'essentiel est acquis, mais il faut encore consolider les bases."
#    else:
#        comm_final += f"Des difficultés persistent, {il_elle.lower()} doit être davantage soutenu{e}."

#    return comm_final

def get_comment_bank(eleve):
    """Retourne une banque de segments personnalisés avec le nom de l'élève."""
    return {
        "intro": [
            f"Un bilan très positif pour {eleve}. ", f"C'est un travail de qualité, {eleve}. ", 
            f"On observe une implication réelle chez {eleve}. ", f"Un ensemble sérieux pour {eleve}. ",
            f"Résultats encourageants concernant {eleve}. ", f"Travail appliqué de la part de {eleve}. ",
            f"Une bonne saisie des notions par {eleve}. ", f"L'investissement de {eleve} se confirme. "
        ],
        "delta_plus": [
            f"Quelle belle progression, {eleve} ! ", f"Les efforts de {eleve} portent leurs fruits. ",
            f"Une dynamique ascendante très motivante pour {eleve}. ", f"Bravo {eleve} pour ce gain de confiance ! ",
            f"Une montée en puissance très appréciable chez {eleve}. ", f"Les progrès de {eleve} sont flagrants. ",
            f"{eleve} gravit les échelons avec succès. ", f"Un saut qualitatif impressionnant pour {eleve} ! "
        ],
        "delta_moins": [
            f"Une petite baisse pour {eleve} ce coup-ci. ", f"{eleve} est un peu en retrait, restons mobilisés. ",
            f"Un coup de mou passager pour {eleve}, ne baisse pas les bras. ", f"{eleve} doit se remobiliser. ",
            f"Attention {eleve}, les bases doivent être revues. ", f"Un score qui invite {eleve} à plus de vigilance. ",
            f"Ce DS était complexe pour {eleve}, on analyse l'erreur ensemble. "
        ],
        "delta_stable": [
            f"Des résultats très stables pour {eleve}. ", f"La régularité est au rendez-vous chez {eleve}. ", 
            f"{eleve} maintient un niveau constant. ", f"Le travail de {eleve} reste solide et régulier. "
        ],
        "premier_ds": [
            f"Premier bilan pour {eleve} dans cette matière. ", f"Une première évaluation prometteuse pour {eleve}. ",
            f"Un point de départ intéressant pour {eleve}. ", f"Début des apprentissages validé pour {eleve}. "
        ],
        "excellent": [
            f"{eleve} montre une maîtrise remarquable.", f"Le travail de {eleve} est d'une précision exemplaire.", 
            f"Les acquis de {eleve} sont parfaitement solides.", f"C'est un sans-faute pour {eleve} !", 
            f"{eleve} a une compréhension totale du sujet."
        ],
        "bien": [
            f"{eleve} mène bien son travail.", f"Une bonne autonomie de {eleve} sur ces notions.",
            f"C'est très satisfaisant, {eleve} doit continuer ainsi.", f"Belle réussite de {eleve} sur ce devoir."
        ],
        "moyen": [
            f"{eleve} a compris l'essentiel, mais des détails échappent encore.", f"Un résultat correct que {eleve} devra consolider.",
            f"Attention aux étourderies, {eleve} y est presque.", f"Des efforts de concentration aideront {eleve}."
        ],
        "difficile": [
            f"Des difficultés persistent pour {eleve}, un soutien est recommandé.", f"Les notions ne sont pas encore stabilisées chez {eleve}.",
            f"{eleve} a besoin de manipuler davantage pour comprendre.", f"{eleve} doit reprendre les bases avec attention."
        ],
        "fin": [
            f" Continue ainsi, {eleve} !", f" Je te félicite, {eleve} !", f" Quel beau parcours, {eleve} !", 
            f" Bravo {eleve} !", f" On lâche rien, {eleve} !", f" Je crois en tes capacités, {eleve} !"
        ]
    }


def get_smart_comment(eleve, note, matiere):
    global df
    if note == "Absent" or not eleve or not matiere: 
        return "Absent(e) lors de l'évaluation."
    
    try: n_actuelle = float(note)
    except: return ""

    # --- CALCUL DU DELTA ---
    delta = None
    prec_df = df[(df["eleve"] == eleve) & (df["matiere"] == matiere) & (df["note"] != "Absent")].copy()
    if not prec_df.empty:
        prec_df["date"] = pd.to_datetime(prec_df["date"])
        prec_df = prec_df.sort_values("date")
        derniere_note = float(prec_df.iloc[-1]["note"])
        delta = n_actuelle - derniere_note

    # --- RÉCUPÉRATION DE LA BANQUE PERSONNALISÉE ---
    bank = get_comment_bank(eleve)

    # --- ASSEMBLAGE ---
    final_txt = ""
    
    # 1. Introduction (1 fois sur 3 pour ne pas être trop lourd)
    if random.random() > 0.66:
        final_txt += random.choice(bank["intro"])

    # 2. Dynamique (Delta)
    if delta is not None:
        if delta > 0: final_txt += random.choice(bank["delta_plus"])
        elif delta < 0: final_txt += random.choice(bank["delta_moins"])
        else: final_txt += random.choice(bank["delta_stable"])
    else:
        final_txt += random.choice(bank["premier_ds"])

    # 3. Niveau
    if n_actuelle >= 18: final_txt += random.choice(bank["excellent"])
    elif n_actuelle >= 14: final_txt += random.choice(bank["bien"])
    elif n_actuelle >= 10: final_txt += random.choice(bank["moyen"])
    else: final_txt += random.choice(bank["difficile"])

    # 4. Fin
    final_txt += random.choice(bank["fin"])

    return final_txt


# --- FONCTIONS TECHNIQUES ---

def get_choices(column, filter_col=None, filter_val=None):
    temp_df = load_data()
    if temp_df.empty: return []
    if filter_col and filter_val:
        temp_df = temp_df[temp_df[filter_col] == filter_val]
    return sorted(temp_df[column].dropna().unique().tolist())

def save_plot_to_file(fig):
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    fig.savefig(path, format='png', dpi=300, bbox_inches='tight')
    return path

# --- LOGIQUE DE SUIVI ---

def plot_student_evolution(eleve, matiere):
    global df
    fig, ax = plt.subplots(figsize=(10, 5))
    if not eleve or not matiere:
        return fig, "### Sélectionnez un élève ET une matière", pd.DataFrame(), None

    sub = df[(df["eleve"] == eleve) & (df["matiere"] == matiere)].copy()
    sub_n = sub[sub["note"] != "Absent"].copy()
    sub_n["note"] = pd.to_numeric(sub_n["note"], errors='coerce').dropna()
    
    if sub_n.empty: return fig, f"Aucune note en {matiere}.", pd.DataFrame(), None

    sub_n["date"] = pd.to_datetime(sub_n["date"])
    sub_n = sub_n.sort_values("date")
    
    # 
    ax.plot(sub_n["date"], sub_n["note"], marker="o", color="#4A90E2", linewidth=3, markersize=10, zorder=3)
    
    for x, y in zip(sub_n["date"], sub_n["note"]):
        ax.vlines(x, 0, y, linestyle="--", color="gray", alpha=0.4, zorder=1)
        ax.annotate(f"{y:g}", (x, y), textcoords="offset points", xytext=(0,12), ha='center',
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec="#4A90E2"))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax.set_ylim(0, 22)
    ax.set_title(f"Progression en {matiere} : {eleve}", pad=25, fontweight="bold")
    ax.grid(True, axis='y', alpha=0.2)
    plt.xticks(rotation=45)
    fig.tight_layout()
    
    table_res = sub_n[["date", "devoir", "note", "commentaire"]].sort_values("date", ascending=False)
    table_res["date"] = table_res["date"].dt.strftime('%d/%m/%Y')
    return fig, f"### Analyse de {eleve}", table_res, save_plot_to_file(fig)

def compute_stats(matiere, devoir):
    global df
    fig, ax = plt.subplots(figsize=(10, 6))
    if not matiere or not devoir or df.empty: return "Sélection.", pd.DataFrame(), fig, None
    sub = df[(df["matiere"] == matiere) & (df["devoir"] == devoir)].copy()
    sub_n = sub[sub["note"] != "Absent"].copy()
    sub_n["note"] = pd.to_numeric(sub_n["note"], errors='coerce').dropna()
    
    if sub_n.empty: return "Aucune note.", pd.DataFrame(), fig, None

    stats = f"📊 {matiere} - {devoir}\nMoyenne classe : {sub_n['note'].mean():.2f}/20"
    
    # 
    ax.hist(sub_n["note"], bins=np.arange(0, 22)-0.5, color="#4A90E2", edgecolor="white", rwidth=0.8)
    ax.set_xticks(range(21))
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.grid(True, axis='both', linestyle='--', alpha=0.5)
    
    classement = sub[["eleve", "note", "commentaire"]].sort_values("note", ascending=False)
    return stats, classement, fig, save_plot_to_file(fig)

# --- FONCTION DÉMO ---

def run_full_demo():
    global df
    data = [
        ["Alice", "2025-09-10", "Français", "Dictée 1", "12", "Début de l'année."],
        ["Alice", "2025-10-05", "Français", "Dictée 2", "16", "En gros progrès !"],
        ["Bob", "2025-09-10", "Français", "Dictée 1", "14", "Bien."],
        ["Bob", "2025-10-05", "Français", "Dictée 2", "10", "Attention à la concentration."],
    ]
    df = pd.DataFrame(data, columns=["eleve", "date", "matiere", "devoir", "note", "commentaire"])
    save_data(df)
    return (
        "✅ Mode Démo activé (Alice et Bob chargés)",
        gr.update(choices=["Alice", "Bob"]),
        gr.update(choices=["Français", "Mathématiques"]),
        df
    )
    
def reset_to_empty():
    global df
    # On crée un tableau vide avec les bonnes colonnes
    df = pd.DataFrame(columns=["eleve", "date", "matiere", "devoir", "note", "commentaire"])
    save_data(df) # On écrase le fichier sur le disque/Hub
    return (
        "🗑️ Carnet réinitialisé ! Vous pouvez recommencer à zéro.",
        gr.update(choices=[], value=None), # Vide le menu Élève
        gr.update(choices=[], value=None), # Vide le menu Matière
        df                                # Vide le tableau d'historique
    )
    
def add_grade(eleve, matiere, devoir, note, date_input, commentaire):
    global df
    if not eleve or not matiere or note is None:
        return "⚠️ Erreur : Infos manquantes.", gr.update(), df, None, None

    d = date_input if date_input else datetime.now().date().isoformat()
    new_row = {"eleve": str(eleve).strip(), "date": d, "matiere": str(matiere).strip(),
               "devoir": str(devoir).strip() or "Évaluation", "note": note, "commentaire": str(commentaire).strip()}
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_data(df)
    
    # On génère immédiatement le nouveau graphique pour l'élève en question
    fig, msg, table, file = plot_student_evolution(eleve, matiere)
    
    return "✅ Note enregistrée !", gr.update(choices=get_choices("eleve")), df, fig, table
# --- INTERFACE ---

with gr.Blocks(theme=gr.themes.Soft(), title="Assistant Notes CP") as demo:
    gr.Markdown("# Bienvenu dans ton outil")

    with gr.Tab("📝 Saisie"):
        with gr.Row():
            mat_in = gr.Dropdown(label="📚 Matière", choices=["Français", "Mathématiques", "Éveil", "Sport"], allow_custom_value=True)
            dev_in = gr.Dropdown(label="Nom du Devoir", choices=[], allow_custom_value=True)
        with gr.Row():
            date_in = gr.DateTime(label="📅 Date", include_time=False, type="string")
            note_in = gr.Dropdown(label="⭐ Note", choices=["Absent"] + [str(i) for i in range(21)], allow_custom_value=True)
            #genre_in = gr.Radio(["M", "F"], label="Genre", value="M")
        with gr.Row():
            eleve_in = gr.Dropdown(label="👤 Élève", choices=get_choices("eleve"), allow_custom_value=True, scale=2)
            with gr.Column(scale=3):
                comm_in = gr.Textbox(label="💬 Appréciation")
                # ON AJOUTE LE NOM DE L'ÉLÈVE AUX INPUTS DU BOUTON
                gen_btn = gr.Button("🎲 Inspirer (Analyse Δ)")
        add_btn = gr.Button("💾 Enregistrer", variant="primary")
        status_msg = gr.Markdown()

    with gr.Tab("📈 Suivi Individuel"):
        with gr.Row():
            eleve_sel = gr.Dropdown(label="1. Élève", choices=get_choices("eleve"), scale=2)
            matiere_sel = gr.Dropdown(label="2. Matière", choices=["Français", "Mathématiques"], scale=2)
            dl_indiv = gr.File(label="Télécharger", scale=1)
        plot_out = gr.Plot()
        moy_display = gr.Markdown()
        table_indiv = gr.Dataframe(label="Historique")

    with gr.Tab("📊 Classe"):
        with gr.Row():
            mat_st = gr.Dropdown(label="Matière", choices=["Français", "Mathématiques"])
            dev_st = gr.Dropdown(label="Devoir")
        with gr.Row():
            stats_out = gr.Textbox(label="Bilan")
            hist_out = gr.Plot()
            dl_class = gr.File(label="Télécharger Histogramme")
        rank_table = gr.Dataframe(label="Classement")

    with gr.Tab("💾 Configuration"):
        demo_btn = gr.Button("🚀 CHARGER DONNÉES DÉMO", variant="secondary")
        export_btn = gr.Button("📤 Exporter CSV")
        reset_btn = gr.Button("🗑️ VIDER TOUT LE CARNET", variant="danger")
        file_output = gr.File()
        status_admin = gr.Markdown()

    # --- CALLBACKS ---
    mat_in.change(fn=lambda m: gr.update(choices=get_choices("devoir", "matiere", m)), inputs=mat_in, outputs=dev_in)
    mat_st.change(fn=lambda m: gr.update(choices=get_choices("devoir", "matiere", m)), inputs=mat_st, outputs=dev_st)
    
    gen_btn.click(fn=get_smart_comment, inputs=[eleve_in, note_in, mat_in], outputs=comm_in) # ,genre_in]
    
    #add_btn.click(fn=lambda *args: (f"✅ Enregistré !", gr.update(choices=get_choices("eleve"))), inputs=[eleve_in, mat_in, dev_in, note_in, date_in, comm_in], outputs=[status_msg, eleve_sel])
    add_btn.click(
    fn=add_grade, 
    inputs=[eleve_in, mat_in, dev_in, note_in, date_in, comm_in], 
    outputs=[status_msg, eleve_sel, table_indiv, plot_out, table_indiv] # On met à jour l'onglet Suivi ici !
    )

    eleve_sel.change(fn=plot_student_evolution, inputs=[eleve_sel, matiere_sel], outputs=[plot_out, moy_display, table_indiv, dl_indiv])
    matiere_sel.change(fn=plot_student_evolution, inputs=[eleve_sel, matiere_sel], outputs=[plot_out, moy_display, table_indiv, dl_indiv])
    
    dev_st.change(fn=compute_stats, inputs=[mat_st, dev_st], outputs=[stats_out, rank_table, hist_out, dl_class])
    reset_btn.click(fn=reset_to_empty, outputs=[status_admin, eleve_sel, mat_in, table_indiv])
    export_btn.click(fn=lambda: str(DATA_PATH), outputs=file_output)
    demo_btn.click(fn=run_full_demo, outputs=[status_admin, eleve_sel, matiere_sel, table_indiv])
if __name__ == "__main__":
    demo.launch()
