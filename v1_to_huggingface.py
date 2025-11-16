import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
import numpy as np

DATA_PATH = Path("/Users/gsinger/notesapp/grades.csv")

def load_data():
    """Charge les données depuis grades.csv ou crée un DataFrame vide."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, parse_dates=["date"])
    else:
        return pd.DataFrame(columns=["student", "date", "assignment", "grade"])


# DataFrame global en mémoire
df = load_data()


def get_student_list(df_local: pd.DataFrame):
    if df_local.empty:
        return []
    return sorted(df_local["student"].dropna().unique().tolist())


def add_grade(student, assignment, grade, date_input):
    """Ajoute une note au DataFrame + sauvegarde CSV."""
    global df

    student = (student or "").strip()
    assignment = (assignment or "").strip()

    # Validation ultra simple (on pourra raffiner)
    if not student:
        return "Erreur : le nom de l'élève est obligatoire.", gr.update(), df

    try:
        grade_float = float(grade)
    except Exception:
        return "Erreur : la note doit être un nombre.", gr.update(), df

    try:
        if date_input:
            d = pd.to_datetime(date_input).date()
        else:
            d = date.today()
    except Exception:
        d = date.today()

    new_row = {
        "student": student,
        "date": d.isoformat(),
        "assignment": assignment if assignment else "Contrôle",
        "grade": grade_float,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    students = get_student_list(df)
    msg = f"Note ajoutée pour {student} : {grade_float} ({new_row['assignment']}, {d})."
    return msg, gr.update(choices=students, value=student), df


def refresh_students():
    """Recharge le CSV et met à jour la liste d'élèves et le tableau."""
    global df
    df = load_data()
    students = get_student_list(df)
    return gr.update(choices=students), df


import numpy as np  # en haut du fichier

def plot_student(student):
    """Retourne une figure matplotlib avec l'évolution des notes."""
    global df
    fig, ax = plt.subplots()

    if not student or df.empty:
        ax.set_title("Pas encore de notes")
        ax.set_xlabel("Date")
        ax.set_ylabel("Note")
        return fig

    sub = df[df["student"] == student].copy()
    if sub.empty:
        ax.set_title(f"Aucune note pour {student}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Note")
        return fig

    # Normalisation robuste de la colonne date
    # Si ce n'est pas déjà du datetime64, on convertit
    if not np.issubdtype(sub["date"].dtype, np.datetime64):
        sub["date"] = pd.to_datetime(
            sub["date"],
            format="mixed",      # accepte "2025-10-10" ET "2025-10-10 00:00:00"
            errors="coerce",     # les valeurs bizarres deviennent NaT
        )

    # On enlève les dates foireuses
    sub = sub.dropna(subset=["date"])
    sub = sub.sort_values("date")

    if sub.empty:
        ax.set_title(f"Aucune date valide pour {student}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Note")
        return fig

    ax.plot(sub["date"], sub["grade"], marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Note")
    ax.set_title(f"Évolution des notes - {student}")
    ax.set_ylim(0, 20)  # si les notes sont sur 20

    fig.autofmt_xdate()
    return fig

def get_csv_file():
    """Retourne le chemin vers le CSV pour téléchargement."""
    global df
    # On s'assure que le fichier existe et est à jour
    df.to_csv(DATA_PATH, index=False)
    return str(DATA_PATH)


with gr.Blocks(title="Suivi des notes") as demo:
    gr.Markdown(
        """
        # 📊 Suivi des notes des élèves

        - Ajoute des notes
        - Visualise l'évolution des notes pour chaque élève
        - Exporte toutes les notes au format CSV
        """
    )

    # ---------- Onglet 1 : Ajouter une note ----------
    with gr.Tab("Ajouter une note"):
        with gr.Row():
            student_input = gr.Textbox(
                label="Nom de l'élève",
                placeholder="Ex: Alice Dupont",
            )
            assignment_input = gr.Textbox(
                label="Contrôle / Chapitre",
                placeholder="Ex: Contrôle chapitre 1",
            )
        with gr.Row():
            grade_input = gr.Textbox(
                label="Note",
                placeholder="Ex: 15.5",
            )
            date_input = gr.Textbox(
                label="Date (optionnel, format AAAA-MM-JJ)",
                placeholder="Ex: 2025-11-16",
            )

        add_button = gr.Button("Ajouter la note ✅")
        status_output = gr.Textbox(label="Statut", interactive=False)

    # ---------- Onglet 2 : Visualiser ----------
    with gr.Tab("Visualiser les notes"):
        refresh_button = gr.Button("🔄 Recharger les données")
        student_dropdown = gr.Dropdown(
            label="Choisir un élève",
            choices=get_student_list(df),
        )

        with gr.Row():
            line_plot = gr.Plot(label="Évolution des notes")

        table = gr.Dataframe(
            label="Toutes les notes",
            value=df,
            interactive=False,
        )

    # ---------- Onglet 3 : Exporter ----------
    with gr.Tab("Exporter les notes"):
        gr.Markdown(
            "### 📥 Exporter les notes\n"
            "Clique sur le bouton ci-dessous pour récupérer le fichier CSV "
            "contenant toutes les notes."
        )
        export_button = gr.Button("Préparer le fichier CSV")
        download_file = gr.File(label="Télécharger grades.csv")

    # ---------- Callbacks (TOUS à l'intérieur du Blocks) ----------

    add_button.click(
        fn=add_grade,
        inputs=[student_input, assignment_input, grade_input, date_input],
        outputs=[status_output, student_dropdown, table],
    )

    refresh_button.click(
        fn=refresh_students,
        inputs=None,
        outputs=[student_dropdown, table],
    )

    student_dropdown.change(
        fn=plot_student,
        inputs=student_dropdown,
        outputs=line_plot,
    )

    export_button.click(
        fn=get_csv_file,
        inputs=None,
        outputs=download_file,
    )

if __name__ == "__main__":
    demo.launch()

