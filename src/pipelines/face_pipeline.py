import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st

from src.database.db import get_all_students


@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector()

    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec


def get_face_embeddings(image_np):
    detector, sp, facerec = load_dlib_models()
    faces = detector(image_np, 1)

    encodings = []

    for face in faces:
        shape = sp(image_np, face)
        # num_jitters=1 keeps embeddings consistent with stored enrollment data.
        # Changing this would make login embeddings incompatible with stored ones.
        face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1)
        encodings.append(np.array(face_descriptor))

    return encodings


@st.cache_resource
def get_trained_model():
    X = []
    y = []

    student_db = get_all_students()

    if not student_db:
        return None

    for student in student_db:
        embedding = student.get('face_embedding')
        if embedding:
            X.append(np.array(embedding))
            y.append(student.get('student_id'))

    if len(X) == 0:
        return 0

    clf = SVC(kernel='linear', probability=True, class_weight='balanced')

    try:
        clf.fit(X, y)
    except ValueError:
        pass

    return {'clf': clf, 'X': X, 'y': y}


def train_classifier():
    st.cache_resource.clear()
    model_data = get_trained_model()
    return bool(model_data)


def predict_attendance(class_image_np):
    encodings = get_face_embeddings(class_image_np)

    detected_student = {}

    model_data = get_trained_model()

    if not model_data:
        return detected_student, [], len(encodings)

    clf = model_data['clf']
    X_train = model_data['X']
    y_train = model_data['y']

    all_students = sorted(list(set(y_train)))

    # ── Distance threshold ───────────────────────────────────────────────────
    # Euclidean distance between 128-d dlib face embeddings.
    #   0.60 = original default (too loose — caused friends to match each other)
    #   0.50 = strict but fair for single-photo enrollment
    #          Your own face from a different angle/light → typically 0.30–0.47
    #          A different person                         → typically 0.55–0.80
    # NOTE: Do NOT add an SVC confidence gate here. With only 1 training sample
    # per class, sklearn's Platt-scaling probabilities are unreliable and will
    # incorrectly reject valid users.
    DISTANCE_THRESHOLD = 0.50
    # ────────────────────────────────────────────────────────────────────────

    for encoding in encodings:
        # Step 1: Use SVC to predict the most likely student
        if len(all_students) >= 2:
            predicted_id = int(clf.predict([encoding])[0])
        else:
            predicted_id = int(all_students[0])

        # Step 2: Verify by computing Euclidean distance against ALL stored
        # embeddings for that student, and take the closest (best) match.
        # This correctly handles students with multiple enrolled photos.
        candidate_embeddings = [
            X_train[i] for i, sid in enumerate(y_train) if sid == predicted_id
        ]

        if not candidate_embeddings:
            continue

        best_distance = min(
            np.linalg.norm(stored_emb - encoding)
            for stored_emb in candidate_embeddings
        )

        if best_distance <= DISTANCE_THRESHOLD:
            detected_student[predicted_id] = True

    return detected_student, all_students, len(encodings)
