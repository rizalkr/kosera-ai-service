# Timpa README.md dengan konfigurasi Metadata
cat <<EOF > README.md
---
title: Kosera AI Service
emoji: 🚀
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Kosera AI Service
Backend AI untuk Sistem Pendukung Keputusan Kosera.
Running on Docker + FastAPI + Sentence Transformers.
EOF