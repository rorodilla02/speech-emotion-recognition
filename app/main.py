"""
Prototipe Deteksi Emosi Berbasis Suara - Antarmuka Streamlit
Jalur 1 subbab 3.7

LOKASI TARGET DI REPO: app/main.py
Membutuhkan juga: .streamlit/config.toml (lihat berkas terpisah) untuk
mengganti primaryColor bawaan Streamlit yang berwarna merah.

STATUS: alur 3-state (input -> processing -> result/error), tema visual
langit/awan yang menyesuaikan emosi hasil prediksi, playback audio sebelum
dan sesudah analisis, serta inferensi lewat src/ser/inference
(AudioBytesLoader + EmotionPredictor) yang memakai ulang AudioPreprocessor
dan FeatureExtractor apa adanya (mitigasi R-05, train-serve mismatch).

REVISI PEMBIMBING (putaran 1):
1. Warna kontrol interaktif tidak lagi merah (lewat config.toml), dan
   seluruh latar hasil dibuat terang agar kontras teks memenuhi ambang
   keterbacaan WCAG 2.1 AA (rasio >= 4.5:1 untuk teks normal).
2. Ukuran font dasar diperbesar; blok keterangan pembatas memakai
   keluarga font serif agar berbeda secara visual dari teks biasa.
3. Playback audio tersedia sebelum dan sesudah analisis.

BELUM DIVERIFIKASI - lakukan sebelum pengujian fungsional:
1. Decode MP3 dari BytesIO (bukan path) - lihat catatan di
   audio_bytes_loader.py. Tes manual dengan berkas .mp3 asli.
2. MODEL_PATH relatif terhadap direktori kerja saat `streamlit run`
   dijalankan - pastikan dijalankan dari root repo.
3. `keras.models.load_model` vs `tf.keras.models.load_model` - kode ini
   pakai `import keras` langsung sesuai stack Keras 3.15 mandiri.
"""

import time

import soundfile as sf
import streamlit as st

from ser.inference.audio_bytes_loader import AudioBytesLoader
from ser.inference.predictor import EmotionPredictor

MODEL_PATH = "data/models/rm1/seed_42/best_model.keras"

EMOTION_LABELS_EN = ("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")
LABEL_ID = {
    "Angry": "Marah",
    "Disgust": "Jijik",
    "Fear": "Takut",
    "Happy": "Senang",
    "Neutral": "Netral",
    "Sad": "Sedih",
    "Surprise": "Terkejut",
}
ALLOWED_EXTENSIONS = ("wav", "flac", "ogg", "mp3")
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 8.0
IDEAL_DURATION_RANGE = (3.0, 5.0)

# 1/7 = 0.143 adalah chance level tujuh kelas. Di atas chance tapi belum
# dominan -> tema divisualkan pudar, bukan tema penuh.
MASCOT_CONFIDENT_THRESHOLD = 0.35

MASCOT_PLACEHOLDER = {
    "Senang": {"confident": "☀️", "unsure": "🌤️"},
    "Sedih": {"confident": "🌧️", "unsure": "⛅"},
    "Marah": {"confident": "⛈️", "unsure": "⛅"},
    "Takut": {"confident": "🌫️", "unsure": "⛅"},
    "Jijik": {"confident": "🌪️", "unsure": "⛅"},
    "Netral": {"confident": "⛅", "unsure": "⛅"},
    "Terkejut": {"confident": "🌈", "unsure": "⛅"},
}

# Palet batang distribusi. Hue mengikuti asosiasi warna-emosi yang lazim
# (merah-jingga = kemarahan, biru = kesedihan, kuning = kegembiraan,
# ungu = ketakutan, hijau = kejijikan, merah muda = keterkejutan,
# abu kebiruan = netral), tetapi seluruhnya dijaga pada tingkat
# kecerahan tinggi agar teks gelap di atasnya tetap terbaca.
# Warna TIDAK dipakai sebagai satu-satunya pembawa makna: setiap batang
# selalu didampingi label teks dan angka persentase.
PASTEL_PALETTE = {
    "Marah": "#E8867A",
    "Jijik": "#9DBE7E",
    "Takut": "#A48FC8",
    "Senang": "#F2C14E",
    "Netral": "#8FB0C4",
    "Sedih": "#7C9BC9",
    "Terkejut": "#DE8FC0",
}

# Seluruh gradien latar dibuat TERANG. Versi sebelumnya memakai gradien
# gelap (mis. Marah #6B4A5A) yang membuat teks gelap di atasnya tidak
# terbaca. Teks aplikasi memakai #2B2B2B; terhadap warna-warna di bawah
# ini rasio kontrasnya berada di atas ambang WCAG 2.1 AA (4.5:1).
DEFAULT_THEME = {
    "gradient": "linear-gradient(180deg, #AED9F2 0%, #EAF6FF 100%)",
    "cloud": "rgba(255,255,255,0.92)",
}
MOOD_THEMES = {
    None: DEFAULT_THEME,
    "Senang": {
        "gradient": "linear-gradient(180deg, #FFE9B0 0%, #FFF9E8 100%)",
        "gradient_muted": "linear-gradient(180deg, #F2ECD9 0%, #FAF7EE 100%)",
        "cloud": "rgba(255,255,255,0.95)",
        "cloud_muted": "rgba(255,255,255,0.8)",
    },
    "Sedih": {
        "gradient": "linear-gradient(180deg, #C5D4E8 0%, #EEF3FA 100%)",
        "gradient_muted": "linear-gradient(180deg, #DCE3ED 0%, #F5F8FC 100%)",
        "cloud": "rgba(255,255,255,0.9)",
        "cloud_muted": "rgba(255,255,255,0.78)",
    },
    "Marah": {
        "gradient": "linear-gradient(180deg, #F8C9C0 0%, #FDEEEA 100%)",
        "gradient_muted": "linear-gradient(180deg, #EFD9D4 0%, #FAF1EF 100%)",
        "cloud": "rgba(255,255,255,0.92)",
        "cloud_muted": "rgba(255,255,255,0.8)",
    },
    "Takut": {
        "gradient": "linear-gradient(180deg, #D6CDE8 0%, #F3EFFA 100%)",
        "gradient_muted": "linear-gradient(180deg, #E4DFEE 0%, #F8F6FC 100%)",
        "cloud": "rgba(255,255,255,0.9)",
        "cloud_muted": "rgba(255,255,255,0.78)",
    },
    "Jijik": {
        "gradient": "linear-gradient(180deg, #CFE0BC 0%, #F0F6E8 100%)",
        "gradient_muted": "linear-gradient(180deg, #E0E8D6 0%, #F6F9F1 100%)",
        "cloud": "rgba(255,255,255,0.92)",
        "cloud_muted": "rgba(255,255,255,0.8)",
    },
    "Netral": {
        "gradient": "linear-gradient(180deg, #D8E3EA 0%, #F3F7F9 100%)",
        "gradient_muted": "linear-gradient(180deg, #E4EBEF 0%, #F7FAFB 100%)",
        "cloud": "rgba(255,255,255,0.92)",
        "cloud_muted": "rgba(255,255,255,0.8)",
    },
    "Terkejut": {
        "gradient": "linear-gradient(180deg, #F7CFE6 0%, #FDEFF7 100%)",
        "gradient_muted": "linear-gradient(180deg, #EDDCE6 0%, #FAF2F6 100%)",
        "cloud": "rgba(255,255,255,0.94)",
        "cloud_muted": "rgba(255,255,255,0.82)",
    },
}


def inject_theme_css(mood_label: str | None, muted: bool = False, animate: bool = True):
    """Menyuntikkan latar langit + awan CSS murni (tanpa aset gambar),
    sekaligus aturan tipografi.

    animate: True di halaman input dan processing (awan bergerak pelan),
    False di halaman result dan error (awan diam, fokus ke konten)."""
    theme = MOOD_THEMES.get(mood_label, DEFAULT_THEME)
    gradient = theme.get("gradient_muted" if muted else "gradient", theme.get("gradient", DEFAULT_THEME["gradient"]))
    cloud_color = theme.get("cloud_muted" if muted else "cloud", theme.get("cloud", DEFAULT_THEME["cloud"]))
    animation_value = "drift 40s linear infinite" if animate else "none"

    st.markdown(
        f"""
        <style>
        /* Streamlit memakai satuan rem secara internal, sehingga mengubah
        font-size pada elemen html menskalakan hampir seluruh antarmuka
        sekaligus - lebih tahan perubahan versi daripada menimpa ukuran
        font tiap komponen satu per satu. Default browser 16px. */
        html {{ font-size: 19px; }}

        .stApp {{
            background: {gradient};
            transition: background 1.2s ease-in-out;
            color: #2B2B2B;
        }}

        .stApp h1 {{ font-size: 2.3rem; }}
        .stApp h2, .stApp h3 {{ font-size: 1.6rem; }}
        .stApp p, .stApp label, .stApp li {{ font-size: 1rem; line-height: 1.6; }}
        .stApp small {{ font-size: 0.85rem; }}

        /* Tab label diperbesar dan diberi bobot supaya status aktif
        terbaca dari bentuk teks, tidak hanya dari warna. */
        .stTabs button p {{ font-size: 1.05rem; font-weight: 600; }}

        .stButton button {{ font-size: 1.05rem; padding: 0.6rem 1.4rem; }}

        .sky-cloud {{
            position: fixed;
            z-index: -1;
            pointer-events: none;
            background: {cloud_color};
            border-radius: 50%;
            filter: blur(0.5px);
            animation: {animation_value};
        }}
        .sky-cloud::before, .sky-cloud::after {{
            content: "";
            position: absolute;
            background: inherit;
            border-radius: 50%;
        }}
        .c1 {{ width: 140px; height: 46px; top: 8%; left: -10%; animation-duration: 55s; }}
        .c1::before {{ width: 70px; height: 70px; top: -30px; left: 20px; }}
        .c1::after {{ width: 50px; height: 50px; top: -18px; left: 80px; }}

        .c2 {{ width: 110px; height: 38px; top: 22%; left: 60%; animation-duration: 65s; animation-delay: -20s; }}
        .c2::before {{ width: 56px; height: 56px; top: -24px; left: 14px; }}
        .c2::after {{ width: 40px; height: 40px; top: -14px; left: 64px; }}

        .c3 {{ width: 160px; height: 50px; top: 4%; left: 35%; animation-duration: 75s; animation-delay: -40s; }}
        .c3::before {{ width: 80px; height: 80px; top: -34px; left: 24px; }}
        .c3::after {{ width: 56px; height: 56px; top: -20px; left: 92px; }}

        @keyframes drift {{
            from {{ transform: translateX(0); }}
            to {{ transform: translateX(60vw); }}
        }}

        /* Batang distribusi berbentuk pil */
        .pill-row {{ display: flex; align-items: center; gap: 12px; margin: 9px 0; }}
        .pill-label {{ width: 100px; font-size: 0.95rem; color: #2B2B2B; text-align: right; }}
        .pill-track {{ flex: 1; background: rgba(255,255,255,0.7); border-radius: 999px;
                        height: 28px; overflow: hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.1); }}
        .pill-fill {{ height: 100%; border-radius: 999px; transition: width 0.9s ease-out; }}
        .pill-pct {{ width: 54px; font-size: 0.9rem; color: #2B2B2B; font-weight: 600; }}

        /* Keterangan pembatas. Latar solid supaya terbaca di atas latar
        langit tema apa pun, dan keluarga font serif supaya secara visual
        jelas berbeda dari teks antarmuka lainnya. */
        .disclaimer-banner {{
            background: #FFF3C4;
            border: 2px solid #A8761E;
            border-left: 7px solid #A8761E;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 18px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.14);
        }}
        .disclaimer-banner p {{
            font-family: Georgia, "Times New Roman", serif;
            font-style: italic;
            color: #4A3600;
            font-size: 0.95rem;
            margin: 0;
            line-height: 1.65;
        }}

        .usage-steps {{
            background: rgba(255,255,255,0.85);
            border-radius: 10px;
            padding: 16px 22px;
            margin: 12px 0 20px 0;
        }}
        .usage-steps p {{ color: #2B2B2B; font-size: 1rem; margin: 6px 0; }}

        /* Baris hasil: maskot dan label disejajarkan secara vertikal lewat
        flexbox dalam satu blok, bukan lewat dua kolom Streamlit terpisah.
        Dengan st.columns, isi tiap kolom rata atas, sehingga label teks
        menggantung di puncak maskot yang jauh lebih tinggi. */
        .result-header {{
            display: flex;
            align-items: center;
            gap: 26px;
            margin: 10px 0 20px 0;
        }}
        .result-mascot {{
            font-size: 88px;
            line-height: 1;
            /* flex-shrink: 0 mencegah maskot ikut mengecil ketika label
            di sebelahnya diperbesar - keduanya diatur terpisah. */
            flex-shrink: 0;
        }}
        /* Selector diawali .stApp agar spesifisitasnya (0,2,0) melampaui
        aturan `.stApp p` (0,1,1) di atas. Tanpa ini, ukuran 1rem dari
        aturan paragraf umum yang menang, dan label hasil tampil kecil. */
        .stApp .result-label {{
            font-size: 3rem;
            font-weight: 800;
            color: #2B2B2B;
            margin: 0;
            line-height: 1.1;
            letter-spacing: -0.5px;
        }}
        </style>
        <div class="sky-cloud c1"></div>
        <div class="sky-cloud c2"></div>
        <div class="sky-cloud c3"></div>
        """,
        unsafe_allow_html=True,
    )


def render_pill_bar_chart(probabilities: dict):
    """Batang pil untuk distribusi probabilitas, diurutkan menurun agar
    kelas dominan langsung terlihat."""
    ordered = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    rows_html = ""
    for label, prob in ordered:
        pct = prob * 100
        color = PASTEL_PALETTE.get(label, "#CCCCCC")
        rows_html += f"""
        <div class="pill-row">
            <div class="pill-label">{label}</div>
            <div class="pill-track">
                <div class="pill-fill" style="width:{pct:.1f}%; background:{color};"></div>
            </div>
            <div class="pill-pct">{pct:.0f}%</div>
        </div>
        """
    st.markdown(rows_html, unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "view": "input",
        "audio_bytes": None,
        "audio_name": None,
        "predicted_label": None,
        "probabilities": None,
        "processing_time_sec": None,
        "error_message": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_to_input():
    st.session_state.view = "input"
    st.session_state.audio_bytes = None
    st.session_state.predicted_label = None
    st.session_state.probabilities = None
    st.session_state.error_message = None


def render_disclaimer_banner():
    # Ditampilkan permanen di seluruh state, bukan modal yang bisa ditutup.
    st.markdown(
        """
        <div class="disclaimer-banner">
            <p>
            Ini purwarupa penelitian, bukan alat diagnostik atau klinis.
            Model dilatih pada korpus berbahasa Inggris. Performa pada input
            berbahasa Indonesia mendekati tingkat kebetulan, sehingga hasil
            prediksi tidak dapat dimaknai sebagai penilaian emosi yang akurat.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_usage_instructions():
    st.markdown(
        """
        <div class="usage-steps">
            <p><b>Cara pakai:</b></p>
            <p>1. Pilih salah satu: unggah berkas audio atau rekam suara langsung.</p>
            <p>2. Pastikan durasi 1-8 detik (paling pas 3-5 detik).</p>
            <p>3. Dengarkan dulu hasil rekaman atau berkas yang dipilih lewat pemutar audio.</p>
            <p>4. Tekan tombol "Analisis Emosi", lalu tunggu beberapa detik.</p>
            <p>5. Label emosi dan grafik probabilitasnya muncul otomatis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def validate_audio(uploaded_file) -> tuple[bool, str]:
    """Validasi format dan durasi. Baca metadata saja, tanpa decode penuh
    sinyal, supaya validasi tetap cepat untuk input yang nantinya ditolak."""
    name = uploaded_file.name.lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        return False, (
            f"Format '.{ext}' tidak didukung. Gunakan salah satu dari: "
            f"{', '.join(ALLOWED_EXTENSIONS).upper()}."
        )

    try:
        uploaded_file.seek(0)
        info = sf.info(uploaded_file)
        duration = info.duration
    except Exception:
        return False, "Berkas tidak dapat dibaca atau rusak. Coba unggah berkas lain."

    if not (MIN_DURATION_SEC <= duration <= MAX_DURATION_SEC):
        return False, (
            f"Durasi audio {duration:.1f} detik di luar rentang "
            f"{MIN_DURATION_SEC:.0f}-{MAX_DURATION_SEC:.0f} detik. "
            f"Rentang ideal: {IDEAL_DURATION_RANGE[0]:.0f}-{IDEAL_DURATION_RANGE[1]:.0f} detik."
        )

    return True, ""


@st.cache_resource
def get_predictor() -> EmotionPredictor:
    # Model dimuat sekali per sesi server, bukan tiap kali tombol ditekan.
    return EmotionPredictor(model_path=MODEL_PATH)


def run_inference_pipeline(audio_bytes: bytes) -> tuple[str, dict, float]:
    """Jalur inferensi: AudioBytesLoader (baca dari memori) ->
    EmotionPredictor (AudioPreprocessor -> FeatureExtractor -> model CNN
    RM1). Label Inggris hasil predictor diterjemahkan ke Bahasa Indonesia
    di lapisan UI ini."""
    start = time.perf_counter()

    loader = AudioBytesLoader()
    audio_data = loader.load(audio_bytes)

    predictor = get_predictor()
    result = predictor.predict(audio_data)

    probabilities = {
        LABEL_ID[label]: prob for label, prob in result.probabilities.items()
    }
    predicted_label = LABEL_ID[result.label]

    elapsed = time.perf_counter() - start
    return predicted_label, probabilities, elapsed


def render_input_view():
    inject_theme_css(None)
    render_disclaimer_banner()
    st.title("Deteksi Emosi Berbasis Suara")
    render_usage_instructions()

    tab_upload, tab_record = st.tabs(["Unggah Berkas", "Rekam Suara"])

    uploaded_file = None
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Pilih berkas audio (WAV, FLAC, OGG, MP3)",
            type=list(ALLOWED_EXTENSIONS),
        )
        st.caption(
            f"Durasi {MIN_DURATION_SEC:.0f}-{MAX_DURATION_SEC:.0f} detik "
            f"(ideal {IDEAL_DURATION_RANGE[0]:.0f}-{IDEAL_DURATION_RANGE[1]:.0f} detik)."
        )

    with tab_record:
        recorded = st.audio_input("Rekam suara langsung")
        if recorded is not None:
            uploaded_file = recorded
            uploaded_file.name = "rekaman.wav"

    # Playback sebelum analisis. Bytes dibaca sekali di sini lalu dipakai
    # ulang saat tombol ditekan, supaya berkas tidak dibaca dua kali.
    audio_bytes = None
    if uploaded_file is not None:
        uploaded_file.seek(0)
        audio_bytes = uploaded_file.read()
        uploaded_file.seek(0)

        st.markdown("**Putar audio yang akan dianalisis:**")
        st.audio(audio_bytes)

    if st.button("Analisis Emosi", type="primary", disabled=uploaded_file is None):
        is_valid, message = validate_audio(uploaded_file)
        if not is_valid:
            st.session_state.error_message = message
            st.session_state.view = "error"
        else:
            st.session_state.audio_bytes = audio_bytes
            st.session_state.audio_name = uploaded_file.name
            st.session_state.view = "processing"
        st.rerun()


def render_processing_view():
    inject_theme_css(None)
    render_disclaimer_banner()
    st.title("Memproses...")
    with st.spinner("Menganalisis suara Anda"):
        try:
            label, probabilities, elapsed = run_inference_pipeline(
                st.session_state.audio_bytes
            )
        except Exception:
            st.session_state.error_message = (
                "Gagal memproses audio. Berkas mungkin rusak secara internal "
                "meski lolos validasi awal."
            )
            st.session_state.view = "error"
            st.rerun()
            return

    st.session_state.predicted_label = label
    st.session_state.probabilities = probabilities
    st.session_state.processing_time_sec = elapsed
    st.session_state.view = "result"
    st.rerun()


def render_result_view():
    label = st.session_state.predicted_label
    probs = st.session_state.probabilities
    top_prob = probs[label]
    is_confident = top_prob >= MASCOT_CONFIDENT_THRESHOLD
    mascot = MASCOT_PLACEHOLDER[label]["confident" if is_confident else "unsure"]

    inject_theme_css(label, muted=not is_confident, animate=False)
    render_disclaimer_banner()
    st.title("Hasil Prediksi")

    # Maskot dan label dirender dalam satu blok flex agar sejajar vertikal.
    st.markdown(
        f"""
        <div class="result-header">
            <div class="result-mascot">{mascot}</div>
            <div class="result-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not is_confident:
        st.caption(
            "Distribusi probabilitas cukup merata - model tidak dominan "
            "condong ke satu kelas untuk input ini."
        )

    # Playback sesudah analisis, supaya pengguna bisa mencocokkan hasil
    # dengan audio yang barusan diproses.
    if st.session_state.audio_bytes is not None:
        st.markdown("**Audio yang dianalisis:**")
        st.audio(st.session_state.audio_bytes)

    st.markdown("**Distribusi probabilitas keluaran model** (bukan tingkat keyakinan)")
    render_pill_bar_chart(probs)

    st.caption(f"Waktu proses: {st.session_state.processing_time_sec:.2f} detik")

    if st.button("Analisis Suara Lain"):
        reset_to_input()
        st.rerun()


def render_error_view():
    inject_theme_css(None, animate=False)
    render_disclaimer_banner()
    st.title("Terjadi Kesalahan")
    st.error(st.session_state.error_message)
    if st.button("Coba Lagi"):
        reset_to_input()
        st.rerun()


def main():
    st.set_page_config(page_title="Deteksi Emosi Berbasis Suara", page_icon="☁️")
    init_session_state()

    view = st.session_state.view
    if view == "input":
        render_input_view()
    elif view == "processing":
        render_processing_view()
    elif view == "result":
        render_result_view()
    elif view == "error":
        render_error_view()
    else:
        reset_to_input()
        st.rerun()


if __name__ == "__main__":
    main()