import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
import numpy as np
import tempfile

# Fichier CSV stocké dans le dossier du Space (/app/grades.csv)
DATA_PATH = Path("grades.csv")

def download_plot(student):
    """Génère un fichier PNG du graphique et le retourne pour téléchargement."""
    fig = plot_student(student)  # on récupère la figure existante
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
    return tmp.name


def load_data():
    """Charge les données depuis grades.csv ou crée un DataFrame vide."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        # Normalisation basique des noms de colonnes pour être robuste
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ("etudiant", "étudiant", "eleve", "élève"):
                col_map[c] = "eleve"
            elif cl in ("matiere", "matière", "devoir"):
                col_map[c] = "devoir"
            elif cl == "note":
                col_map[c] = "note"
            elif cl == "date":
                col_map[c] = "date"
        if col_map:
            df = df.rename(columns=col_map)
        # On garantit la présence des 4 colonnes
        for col in ["eleve", "date", "devoir", "note"]:
            if col not in df.columns:
                df[col] = pd.NA
        return df[["eleve", "date", "devoir", "note"]]
    else:
        return pd.DataFrame(columns=["eleve", "date", "devoir", "note"])


# DataFrame global en mémoire
df = load_data()


def get_eleve_list(df_local: pd.DataFrame):
    if df_local.empty:
        return []
    return sorted(df_local["eleve"].dropna().unique().tolist())


def get_devoir_list(df_local: pd.DataFrame):
    if df_local.empty:
        return []
    return sorted(df_local["devoir"].dropna().unique().tolist())


def add_grade(eleve, devoir, note, date_input):
    """Ajoute une note au DataFrame + sauvegarde CSV."""
    global df

    eleve = (eleve or "").strip()
    devoir = (devoir or "").strip()

    if not eleve:
        return (
            "Erreur : le nom de l'élève est obligatoire.",
            gr.update(),
            df,
            gr.update(),
        )

    try:
        note_float = float(note)
    except Exception:
        return (
            "Erreur : la note doit être un nombre.",
            gr.update(),
            df,
            gr.update(),
        )

    try:
        if date_input:
            d = pd.to_datetime(date_input).date()
        else:
            d = date.today()
    except Exception:
        d = date.today()

    new_row = {
        "eleve": eleve,
        "date": d.isoformat(),
        "devoir": devoir if devoir else "Devoir",
        "note": note_float,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    eleves = get_eleve_list(df)
    devoirs = get_devoir_list(df)

    msg = f"Note ajoutée pour {eleve} : {note_float} ({new_row['devoir']}, {d})."
    return msg, gr.update(choices=eleves, value=eleve), df, gr.update(choices=devoirs)


def refresh_all():
    """Recharge le CSV et met à jour tableau, élèves, devoirs."""
    global df
    df = load_data()
    eleves = get_eleve_list(df)
    devoirs = get_devoir_list(df)
    return df, gr.update(choices=eleves), gr.update(choices=devoirs)


def plot_student(eleve):
    """Retourne une figure matplotlib avec l'évolution des notes d'un élève."""
    global df
    fig, ax = plt.subplots()

    if not eleve or df.empty:
        ax.set_title("Pas encore de notes")
        ax.set_xlabel("Date")
        ax.set_ylabel("Note")
        return fig

    sub = df[df["eleve"] == eleve].copy()
    if sub.empty:
        ax.set_title(f"Aucune note pour {eleve}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Note")
        return fig

    # Conversion robuste de la colonne date
    if not np.issubdtype(sub["date"].dtype, np.datetime64):
        sub["date"] = pd.to_datetime(
            sub["date"],
            format="mixed",
            errors="coerce",
        )

    sub = sub.dropna(subset=["date"])
    sub = sub.sort_values("date")

    if sub.empty:
        ax.set_title(f"Aucune date valide pour {eleve}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Note")
        return fig

    ax.plot(sub["date"], sub["note"], marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Note")
    ax.set_title(f"Évolution des notes - {eleve}")
    ax.set_ylim(0, 20)  # si tu notes sur 20

    fig.autofmt_xdate()
    return fig


def compute_stats(devoir):
    """Calcule la moyenne et le classement pour un devoir."""
    global df
    if not devoir or df.empty:
        empty = pd.DataFrame(columns=["rang", "eleve", "note"])
        return "Pas de données pour ce devoir.", empty

    sub = df[df["devoir"] == devoir].copy()
    if sub.empty:
        empty = pd.DataFrame(columns=["rang", "eleve", "note"])
        return f"Aucune note trouvée pour le devoir « {devoir} ».", empty

    sub = sub.dropna(subset=["note", "eleve"])
    if sub.empty:
        empty = pd.DataFrame(columns=["rang", "eleve", "note"])
        return f"Aucune note exploitable pour le devoir « {devoir} ».", empty

    sub = sub.sort_values("note", ascending=False)
    sub["rang"] = range(1, len(sub) + 1)
    moyenne = sub["note"].mean()
    resume = f"Moyenne pour « {devoir} » : {moyenne:.2f} / 20 ({len(sub)} élèves)."
    classement = sub[["rang", "eleve", "note"]]
    return resume, classement


def import_csv(file_obj):
    """Importe un CSV et remplace les données courantes."""
    global df
    if file_obj is None:
        return "Aucun fichier sélectionné.", df, gr.update(), gr.update()
    try:
        df_new = pd.read_csv(file_obj.name)
    except Exception as e:
        return f"Erreur lors de la lecture du CSV : {e}", df, gr.update(), gr.update()

    # Normalisation des colonnes
    col_map = {}
    for c in df_new.columns:
        cl = c.lower()
        if cl in ("etudiant", "étudiant", "eleve", "élève"):
            col_map[c] = "eleve"
        elif cl in ("matiere", "matière", "devoir"):
            col_map[c] = "devoir"
        elif cl == "note":
            col_map[c] = "note"
        elif cl == "date":
            col_map[c] = "date"
    if col_map:
        df_new = df_new.rename(columns=col_map)

    required = {"eleve", "date", "devoir", "note"}
    if not required.issubset(df_new.columns):
        msg = "Le fichier CSV doit contenir les colonnes : eleve, date, devoir, note."
        return msg, df, gr.update(), gr.update()

    df_new = df_new[["eleve", "date", "devoir", "note"]]
    df = df_new.reset_index(drop=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    eleves = get_eleve_list(df)
    devoirs = get_devoir_list(df)
    status = f"Fichier importé avec succès ({len(df)} lignes)."
    return status, df, gr.update(choices=eleves), gr.update(choices=devoirs)


def get_csv_file():
    """Retourne le chemin vers le CSV pour téléchargement."""
    global df
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return str(DATA_PATH)


with gr.Blocks(title="Suivi des notes") as demo:
    gr.Markdown(
        """
        #Suivi des notes des élèves.
        Bienvenue :) 
        """
    )

    # ---------- Onglet 1 : Ajouter une note ----------
    with gr.Tab("Ajouter une note"):
        with gr.Row():
            eleve_input = gr.Textbox(
                label="Nom de l'élève",
            )
            devoir_input = gr.Textbox(
                label="Devoir / Matière",
            )
        with gr.Row():
            note_input = gr.Textbox(
                label="Note",
            )
            date_input = gr.Textbox(
                label="Date (optionnel, AAAA-MM-JJ)",
            )

        add_button = gr.Button("Ajouter la note ✅")
        status_output = gr.Textbox(label="Statut", interactive=False)


    
    with gr.Tab("Visualiser les notes"):
        refresh_button = gr.Button("🔄 Recharger les données")

        eleve_dropdown = gr.Dropdown(
        label="Choisir un élève",
        choices=get_eleve_list(df),)

        with gr.Row():
            line_plot = gr.Plot(label="Évolution des notes")

        table = gr.Dataframe(
        label="Toutes les notes",
        value=df,
        interactive=False,
    )

    # --- Télécharger le graphique ---
        download_plot_button = gr.Button("📥 Télécharger le graphique")
        download_plot_file = gr.File(label="Télécharger le PNG")

        download_plot_button.click(
        fn=download_plot,
        inputs=eleve_dropdown,
        outputs=download_plot_file,
        )


    # ---------- Onglet 2 : Visualiser ----------
    #with gr.Tab("Visualiser les notes"):
    #    refresh_button = gr.Button("🔄 Recharger les données")
    #    eleve_dropdown = gr.Dropdown(
    #        label="Choisir un élève",
    #        choices=get_eleve_list(df),
    #    )

    #    with gr.Row():
    #        line_plot = gr.Plot(label="Évolution des notes")

    #    table = gr.Dataframe(
    #        label="Toutes les notes",
    #        value=df,
    #        interactive=False,
    #    )

    # ---------- Onglet 3 : Statistiques ----------
    with gr.Tab("Statistiques par devoir"):
        devoir_dropdown = gr.Dropdown(
            label="Choisir un devoir",
            choices=get_devoir_list(df),
        )
        stats_text = gr.Textbox(
            label="Moyenne et informations",
            interactive=False,
        )
        classement_table = gr.Dataframe(
            label="Classement sur ce devoir",
            interactive=False,
        )

    # ---------- Onglet 4 : Données ----------
    with gr.Tab("Données (import / export)"):
        gr.Markdown(
            "### 📂 Importer / exporter les notes\n"
            "- Importer un fichier CSV existant pour charger des notes\n"
            "- Exporter le fichier CSV courant pour le sauvegarder sur votre ordinateur"
        )
        csv_upload = gr.File(label="Importer un fichier CSV", file_types=[".csv"])
        import_button = gr.Button("Charger ce CSV")
        import_status = gr.Textbox(label="Statut de l'import", interactive=False)

        export_button = gr.Button("Préparer le fichier CSV pour téléchargement")
        download_file = gr.File(label="Télécharger grades.csv")

    # ---------- Callbacks ----------
    add_button.click(
        fn=add_grade,
        inputs=[eleve_input, devoir_input, note_input, date_input],
        outputs=[status_output, eleve_dropdown, table, devoir_dropdown],
    )

    refresh_button.click(
        fn=refresh_all,
        inputs=None,
        outputs=[table, eleve_dropdown, devoir_dropdown],
    )

    eleve_dropdown.change(
        fn=plot_student,
        inputs=eleve_dropdown,
        outputs=line_plot,
    )

    devoir_dropdown.change(
        fn=compute_stats,
        inputs=devoir_dropdown,
        outputs=[stats_text, classement_table],
    )

    import_button.click(
        fn=import_csv,
        inputs=csv_upload,
        outputs=[import_status, table, eleve_dropdown, devoir_dropdown],
    )

    export_button.click(
        fn=get_csv_file,
        inputs=None,
        outputs=download_file,
    )

if __name__ == "__main__":
    demo.launch()
