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


def format_df_french(dataframe):
    """Retourne une copie du tableau avec les dates au format français."""
    if dataframe.empty:
        return dataframe
    df_display = dataframe.copy()
    # On transforme la colonne date pour l'affichage
    df_display['date'] = pd.to_datetime(df_display['date']).dt.strftime('%d/%m/%Y')
    return df_display

def generate_templates():
    """Génère des fichiers modèles CSV et Excel pour l'utilisateur."""
    df_template = pd.DataFrame(columns=["eleve", "date", "matiere", "devoir", "note", "coeff","commentaire"])
    # On ajoute une ligne d'exemple pour aider la prof
    df_template.loc[0] = ["Jean Dupont", "2025-01-10", "Français", "Dictée 1", "15", "2", "Très bon travail"]
    
    csv_path = "modele_notes.csv"
    xlsx_path = "modele_notes.xlsx"
    
    df_template.to_csv(csv_path, index=False, encoding='utf-8-sig') # utf-8-sig pour que Excel l'ouvre bien
    df_template.to_excel(xlsx_path, index=False)
    
    return csv_path, xlsx_path

def import_external_file(file):
    """Lit un fichier CSV ou XLSX importé et l'ajoute au carnet actuel."""
    global df
    if file is None:
        return "⚠️ Aucun fichier sélectionné.", gr.update(), df
    
    try:
        # Détection du type de fichier
        if file.name.endswith('.csv'):
            new_data = pd.read_csv(file.name)
        elif file.name.endswith('.xlsx'):
            new_data = pd.read_excel(file.name)
        else:
            return "❌ Format non supporté (utilisez .csv ou .xlsx)", gr.update(), df
        
        # Vérification des colonnes
        required_cols = ["eleve", "date", "matiere", "devoir", "note", "commentaire"]
        if not all(col in new_data.columns for col in required_cols):
            return f"❌ Erreur : Le fichier doit contenir les colonnes : {', '.join(required_cols)}", gr.update(), df
        
        # Fusion avec les données existantes
        df = pd.concat([df, new_data], ignore_index=True)
        save_data(df)
        
        noms_tries = sorted(df["eleve"].unique().tolist())
        return f"✅ Import réussi ! {len(new_data)} notes ajoutées.", gr.update(choices=noms_tries), format_df_french(df)
        
    except Exception as e:
        return f"❌ Erreur lors de la lecture : {str(e)}", gr.update(), format_df_french(df)


def load_data():
    #cols = ["eleve", "date", "matiere", "devoir", "note", "commentaire"]
    cols = ["eleve", "date", "matiere", "devoir", "note", "coeff", "commentaire"]
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



def generate_student_summary(eleve):
    global df
    if not eleve or df.empty: return "### Sélectionnez un élève."
    
    # On prépare les données globales de l'élève
    sub_all = df[df["eleve"] == eleve].copy()
    sub_all["note_num"] = pd.to_numeric(sub_all["note"], errors='coerce')
    sub_all = sub_all.dropna(subset=["note_num"])
    
    if sub_all.empty: return "### Aucune donnée numérique pour cet élève."
    
    subjects = sorted(sub_all["matiere"].unique())
    summary = f"# 📋 Bilan Pédagogique : {eleve}\n"
    summary += f"*Généré le {datetime.now().strftime('%d/%m/%Y')}*\n\n---\n"

    for mat in subjects:
        sub_mat = sub_all[sub_all["matiere"] == mat].sort_values("date")
        
        if len(sub_mat) < 2:
            summary += f"### 📚 {mat}\n* Recul insuffisant (une seule évaluation).\n\n"
            continue

        # --- CALCUL DE LA PENTE (Régression Linéaire) ---
        # On convertit les dates en nombre de jours depuis la 1ère note
        dates_dt = pd.to_datetime(sub_mat['date'])
        x = (dates_dt - dates_dt.min()).dt.days
        y = sub_mat['note_num']
        
        # Pente (slope) = points gagnés ou perdus par jour
        slope, intercept = np.polyfit(x, y, 1)
        
        # On calcule l'évolution sur un mois (30 jours) pour que ce soit parlant
        pente_mensuelle = slope * 30
        
        # Dates de la période
        start_p = dates_dt.min().strftime('%d/%m/%y')
        end_p = dates_dt.max().strftime('%d/%m/%y')

        # --- RÉDACTION DU COMMENTAIRE AUTOMATIQUE ---
        if slope > 0.01:
            statut = f"📈 **Progression constante**"
            conseil = f"Une dynamique très positive avec une hausse moyenne de **{pente_mensuelle:.1f} pts/mois**."
        elif slope < -0.01:
            statut = f"⚠️ **Attention au relâchement**"
            conseil = f"Vigilance nécessaire : on observe une baisse tendancielle de **{abs(pente_mensuelle):.1f} pts/mois**."
        else:
            statut = f"📊 **Résultats stables**"
            conseil = "Le niveau de maîtrise reste constant sur la période."

        summary += f"### 📚 {mat}\n"
        summary += f"* **Période** : du {start_p} au {end_p}\n"
        summary += f"* **Tendance** : {statut}\n"
        summary += f"* **Analyse** : {conseil}\n\n"

    return summary

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
    
    if not matiere or not devoir or df.empty: 
        return "Sélectionnez une matière et un devoir.", pd.DataFrame(), fig, None, "### Moyenne : --"
    
    # 1. Préparation des données numériques
    sub = df[(df["matiere"] == matiere) & (df["devoir"] == devoir)].copy()
    sub["note_num"] = pd.to_numeric(sub["note"], errors='coerce')
    sub_n = sub.dropna(subset=["note_num"])
    notes = sub_n["note_num"]
    
    if notes.empty: 
        return "Aucune note numérique.", pd.DataFrame(), fig, None, "### Moyenne : --"

    # 2. CALCULS STATISTIQUES
    moyenne = notes.mean()
    mediane = notes.median()
    q1 = notes.quantile(0.25)
    q3 = notes.quantile(0.75)

    # 3. IDENTIFICATION DES ÉLÈVES PAR QUANTILES
    # Élèves <= Q1 (Groupe de soutien)
    eleves_q1 = sub_n[sub_n["note_num"] <= q1]["eleve"].tolist()
    # Élèves >= Q3 (Groupe d'excellence)
    eleves_q3 = sub_n[sub_n["note_num"] >= q3]["eleve"].tolist()

    # Formattage des listes pour l'affichage
    txt_q1 = ", ".join(eleves_q1) if eleves_q1 else "Aucun"
    txt_q3 = ", ".join(eleves_q3) if eleves_q3 else "Aucun"

    stats_txt = (f"📊 BILAN : {matiere} - {devoir}\n"
                 f"---------------------------------\n"
                 f"Moyenne : {moyenne:.2f} | Médiane : {mediane:.2f}\n"
                 f"---------------------------------\n"
                 f"🚩 GROUPE DE SOUTIEN (≤ {q1:.1f}/20) :\n"
                 f"{txt_q1}\n\n"
                 f"🌟 GROUPE D'EXCELLENCE (≥ {q3:.1f}/20) :\n"
                 f"{txt_q3}\n"
                 f"---------------------------------\n"
                 f"Copies : {len(notes)} | Absents : {len(sub) - len(notes)}")

    # 4. GRAPHIQUE (Histogramme)
    ax.hist(notes, bins=np.arange(0, 22)-0.5, color="#4A90E2", edgecolor="white", alpha=0.7)
    ax.axvline(moyenne, color='red', linestyle='--', label=f'Moyenne')
    ax.axvline(mediane, color='green', linestyle='-', label=f'Médiane')
    ax.set_xticks(range(21))
    ax.legend()
    
    # 5. CLASSEMENT
    sub_pour_tri = sub.copy()
    sub_pour_tri["note_tri"] = sub_pour_tri["note_num"].fillna(-1)
    classement = (sub_pour_tri.sort_values("note_tri", ascending=False)
                  .drop(columns=["note_tri", "note_num"])[["eleve", "note", "commentaire"]])
    
    return stats_txt, classement, fig, save_plot_to_file(fig), f"## 🏆 Moyenne de Classe : {moyenne:.2f}/20"
    
def run_full_demo():
    global df
    # Liste de 40 élèves avec des homonymes (ex: 2 Alice, 2 Bob, 2 Thomas)
    eleves = [
        "Alice Bernard", "Alice Morel", "Bob Martin", "Bob Petit",
        "Charlie Dubois", "Charlie Leroy", "David Garcia", "Emma Roux",
        "Emma Lefebvre", "Fiona David", "Gabriel Bertrand", "Hugo Vincent",
        "Inès Girard", "Jade Lambert", "Kenzo Fontaine", "Léa Bonnet",
        "Léa Muller", "Manon Faure", "Noah Andre", "Olivia Mercier",
        "Paul Simon", "Paul Dupont", "Quentin Lucas", "Rose Brun",
        "Sacha Clement", "Thomas Robert", "Thomas Meyer", "Ugo Barbier",
        "Victoire Colin", "William Adam", "Xavier Marchand", "Yasmine Duval",
        "Zoé Denis", "Zoé Renard", "Arthur Meunier", "Bastien Lemaire",
        "Clara Perrin", "Diane Roche", "Enzo Hubert", "Faustine Roy"
    ]
    
    matieres = ["Français", "Mathématiques"]
    devoirs = {
        "Français": [("Dictée 1", "2025-09-10"), ("Dictée 2", "2025-10-05")],
        "Mathématiques": [("Calcul", "2025-09-15"), ("Géométrie", "2025-11-01")]
    }
    
    data = []
    for nom in eleves:
        # On définit un profil (moyenne de base) pour que l'élève soit cohérent
        profil_eleve = random.randint(7, 17) 
        
        for matiere in matieres:
            for devoir_nom, date_devoir in devoirs[matiere]:
                # On génère une note autour du profil de l'élève
                note_val = profil_eleve + random.randint(-3, 3)
                note_val = max(0, min(20, note_val))
                
                # Simulation d'absences aléatoires
                if random.random() < 0.05:
                    note_str = "Absent"
                    comm = f"Évaluation non réalisée pour {nom}."
                else:
                    note_str = str(note_val)
                    comm = "Travail régulier."
                
                #data.append([nom, date_devoir, matiere, devoir_nom, note_str, comm])
                data.append([nom, date_devoir, matiere, devoir_nom, note_str, 1, comm]) # Ajout du 1 ici

    df = pd.DataFrame(data, columns=["eleve", "date", "matiere", "devoir", "note", "coeff", "commentaire"]) # Ajout de "coeff"  
    #df = pd.DataFrame(data, columns=["eleve", "date", "matiere", "devoir", "note", "commentaire"])
    save_data(df)
    noms_tries = sorted(eleves)
    return (
        f"✅ Classe de {len(eleves)} élèves chargée avec succès !",
        gr.update(choices=noms_tries), # Les noms seront triés par ordre alphabétique
        gr.update(choices=noms_tries), # POUR eleve_in (Saisie) 
        gr.update(choices=matieres),
        format_df_french(df) )    
def reset_to_empty():
    global df
    # On crée un tableau vide avec les bonnes colonnes
    df = pd.DataFrame(columns=["eleve", "date", "matiere", "devoir", "note", "commentaire"])
    save_data(df) # On écrase le fichier sur le disque/Hub
    return (
        "🗑️ Carnet réinitialisé ! Vous pouvez recommencer à zéro.",
        gr.update(choices=[], value=None), # Vide le menu Élève
        gr.update(choices=[], value=None), # Vide le menu Matière
        format_df_french(df)                                # Vide le tableau d'historique
    )
    
def add_grade(eleve, matiere, devoir, note, date_input, coeff, commentaire):
    global df
    if not eleve or not matiere or note is None:
        return "⚠️ Erreur : Infos manquantes.", gr.update(), df, None, None

    d = date_input if date_input else datetime.now().date().isoformat()
    new_row = {"eleve": str(eleve).strip(), "date": d, "matiere": str(matiere).strip(),
               "devoir": str(devoir).strip() or "Évaluation", "note": note,"coeff": float(coeff) ,  "commentaire": str(commentaire).strip()}
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_data(df)
    
    # On prépare les données pour l'onglet Suivi (Graphique + Analyse)
    fig, msg, table, _ = plot_student_evolution(eleve, matiere)
    
    # ON RENVOIE 7 VALEURS
    # 1: status, 2: menu_eleve, 3: df_pour_saisie, 4: fig, 5: synthese, 6: table_indiv, 7: df_live
    #return "✅ Note enregistrée !", gr.update(choices=get_choices("eleve")), df, fig, msg, table, df
    #return "✅ Note enregistrée !", gr.update(choices=get_choices("eleve")), table, fig, msg, table, df
    # Change la fin du return pour utiliser format_df_french(df)
    return "✅ Note enregistrée !", gr.update(choices=get_choices("eleve")), format_df_french(df), fig, msg, table, format_df_french(df)
# --- INTERFACE ---
with gr.Blocks(theme=gr.themes.Default(), title="Assistant Notes CP") as demo:
    gr.Markdown("#Bonjour :) ")

    with gr.Tab("📖 Guide & Aide"):
        gr.Markdown("""
        # L'Assistant Notes : Votre partenaire pédagogique au quotidien
        
        **Libérez-vous du temps pour l'essentiel en automatisant vos bilans, tout en portant un regard précis sur la réussite de chaque élève. Visualisez instantanément la progression individuelle au fil de chaque évaluation, identifiez les trajectoires de progrès ou les besoins de soutien, et laissez l'Assistant Notes simplifier vos appréciations pour des livrets plus humains et sans effort.**
        
        ---
        ### Vos outils clés
        ⭐ **Suivi Dynamique** : Visualisez l'évolution de chaque élève, évaluation après évaluation (DS), grâce à des graphiques de progression chronologiques par matière.
        
        ⭐ **Aide à la Rédaction** : Générez des commentaires personnalisés qui analysent automatiquement l'écart entre le dernier DS et le précédent ($\Delta$).
        
        ⭐ **Saisie Intuitive** : Enregistrez vos notes, dates et coefficients en un clic avec un registre live pour ne jamais perdre le fil de vos saisies.
        
        ⭐ **Analyse "Conseil de Classe"** : Obtenez une synthèse automatique des points forts et des points de vigilance pour chaque profil d'élève.
        """)
    
    with gr.Tab("📝 Saisie"):
        with gr.Row():
            mat_in = gr.Dropdown(label="📚 Matière", choices=["Français", "Mathématiques", "Éveil", "Sport"], allow_custom_value=True)
            dev_in = gr.Dropdown(label="Nom du Devoir", choices=[], allow_custom_value=True)
        with gr.Row():
            date_in = gr.DateTime(label="📅 Date", include_time=False, type="string")
            note_in = gr.Dropdown(label="⭐ Note", choices=["Absent"] + [str(i) for i in range(21)], allow_custom_value=True)
            #genre_in = gr.Radio(["M", "F"], label="Genre", value="M")
            coeff_in = gr.Slider(minimum=1, maximum=4, step=1, label="Coefficient (Poids du devoir)", value=1)
        with gr.Row():
            eleve_in = gr.Dropdown(label="👤 Élève", choices=get_choices("eleve"), allow_custom_value=True, scale=2)
            with gr.Column(scale=3):
                comm_in = gr.Textbox(label="💬 Appréciation", lines=5)
                # ON AJOUTE LE NOM DE L'ÉLÈVE AUX INPUTS DU BOUTON
                gen_btn = gr.Button("🎲 Propose une appréciation personalisée")
        add_btn = gr.Button("💾 Enregistrer", variant="primary")
        status_msg = gr.Markdown()
        # AJOUT ICI (Le Product Designer valide !)
        gr.Markdown("### 📜 Registre des dernières saisies")
        full_table_display = gr.Dataframe(value=format_df_french(df), label="Carnet de notes complet")

        

    with gr.Tab("📈 Suivi Individuel"):
        with gr.Row():
            eleve_sel = gr.Dropdown(label="1. Élève", choices=get_choices("eleve"), scale=2)
            matiere_sel = gr.Dropdown(label="2. Matière", choices=["Français", "Mathématiques"], scale=2)
        
        with gr.Row():
            # On ajoute la synthèse ici
            with gr.Column(scale=2):
                summary_out = gr.Markdown("### Bilan", container=True)
            with gr.Column(scale=2):
                plot_out = gr.Plot()
                
        table_indiv = gr.Dataframe(label="Historique")

    with gr.Tab("📊 Classe"):
        # Nouveau : Affichage de la moyenne générale de la matière en gros
        moy_matiere_display = gr.Markdown("## 🏆 Moyenne Générale de la matière : --/20")
        
        with gr.Row():
            mat_st = gr.Dropdown(label="Matière", choices=["Français", "Mathématiques"])
            dev_st = gr.Dropdown(label="Devoir")
        with gr.Row():
            #stats_out = gr.Textbox(label="Bilan rapide", lines=5)
            stats_out = gr.Textbox(label="Analyse des groupes de niveau", lines=10)
            hist_out = gr.Plot()
            dl_class = gr.File(label="Télécharger Histogramme")
        rank_table = gr.Dataframe(label="Classement de la classe")
        
    with gr.Tab("💾 Exporter vos données"):
        demo_btn = gr.Button("🚀 CHARGER DONNÉES DÉMO", variant="secondary")
        export_btn = gr.Button("📤 Exporter votre fichier avec les notes")
        reset_btn = gr.Button("🗑️ VIDER TOUT LE CARNET", variant="danger")
        file_output = gr.File()
        status_admin = gr.Markdown()

    with gr.Tab("📂 Modèles & Import de vos données"):
        gr.Markdown("### 1. Télécharger un modèle")
        gr.Markdown("Utilisez ces fichiers comme base pour remplir vos notes sur Excel.")
        with gr.Row():
            btn_gen_template = gr.Button("📄 Générer les modèles")
            tpl_csv = gr.File(label="Modèle CSV")
            tpl_xlsx = gr.File(label="Modèle Excel")
            
        gr.Markdown("---")
        gr.Markdown("### 2. Importer vos notes")
        gr.Markdown("Une fois votre fichier rempli, glissez-le ici pour l'ajouter au carnet.")
        file_import = gr.File(label="Déposer un fichier .csv ou .xlsx", file_types=[".csv", ".xlsx"])
        btn_import = gr.Button("📥 Lancer l'importation", variant="primary")
        import_status = gr.Markdown()
    
    
    # --- CALLBACKS ---
    mat_in.change(fn=lambda m: gr.update(choices=get_choices("devoir", "matiere", m)), inputs=mat_in, outputs=dev_in)
    mat_st.change(fn=lambda m: gr.update(choices=get_choices("devoir", "matiere", m)), inputs=mat_st, outputs=dev_st)
    
    gen_btn.click(fn=get_smart_comment, inputs=[eleve_in, note_in, mat_in], outputs=comm_in) # ,genre_in]
    
    #add_btn.click(fn=lambda *args: (f"✅ Enregistré !", gr.update(choices=get_choices("eleve"))), inputs=[eleve_in, mat_in, dev_in, note_in, date_in, comm_in], outputs=[status_msg, eleve_sel])
    add_btn.click(
    fn=add_grade, 
    inputs=[eleve_in, mat_in, dev_in, note_in, date_in, coeff_in, comm_in], 
    outputs=[status_msg, eleve_sel, table_indiv, plot_out, summary_out, table_indiv, full_table_display]
)
    
    eleve_sel.change(
    fn=lambda e, m: (plot_student_evolution(e, m)[0], generate_student_summary(e), plot_student_evolution(e, m)[2]), 
    inputs=[eleve_sel, matiere_sel], 
    outputs=[plot_out, summary_out, table_indiv])
    #eleve_sel.change(fn=plot_student_evolution, inputs=[eleve_sel, matiere_sel], outputs=[plot_out, moy_display, table_indiv, dl_indiv])
    matiere_sel.change(
    fn=lambda e, m: (plot_student_evolution(e, m)[0], generate_student_summary(e), plot_student_evolution(e, m)[2]), 
    inputs=[eleve_sel, matiere_sel], 
    outputs=[plot_out, summary_out, table_indiv])    
    #dev_st.change(fn=compute_stats, inputs=[mat_st, dev_st], outputs=[stats_out, rank_table, hist_out, dl_class])
    dev_st.change(
        fn=compute_stats, 
        inputs=[mat_st, dev_st], 
        outputs=[stats_out, rank_table, hist_out, dl_class, moy_matiere_display])
    
    #reset_btn.click(fn=reset_to_empty, outputs=[status_admin, eleve_sel, mat_in, table_indiv])
    reset_btn.click(
    fn=reset_to_empty, 
    outputs=[status_admin, eleve_sel, mat_in, full_table_display])
    export_btn.click(fn=lambda: str(DATA_PATH), outputs=file_output)
    
    #demo_btn.click(
    #fn=run_full_demo, 
    #outputs=[status_admin, eleve_sel, eleve_in, matiere_sel, table_indiv])

    demo_btn.click(
    fn=run_full_demo, 
    outputs=[status_admin, eleve_sel, eleve_in, matiere_sel, full_table_display])
    
    # Génération des modèles
    btn_gen_template.click(fn=generate_templates, outputs=[tpl_csv, tpl_xlsx])
    
    # Importation de fichiers
    btn_import.click(
    fn=import_external_file, 
    inputs=[file_import], 
    outputs=[import_status, eleve_sel, full_table_display])


    
if __name__ == "__main__":
    demo.launch()
