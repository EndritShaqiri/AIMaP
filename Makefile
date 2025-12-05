# ================================
# AIMaP Makefile
# ================================

# Virtual environment name
VENV = venv

# Python version required by the project
PYTHON = python3.10

# ================================
# SETUP: Create venv + Install deps
# ================================
setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	@echo "Setup complete. Run 'make run' to start AIMaP."

# ================================
# Run Backend (FastAPI)
# ================================
run-backend:
	$(VENV)/bin/uvicorn AI_thrember.api:app --host 0.0.0.0 --port 8000 --reload

# ================================
# Run Frontend (Static Website)
# ================================
run-frontend:
	cd web && $(PYTHON) -m http.server 8080

# ================================
# Open browser automatically
# ================================
open:
	sleep 2 && \
	( xdg-open http://localhost:8080/index.html || \
	  open http://localhost:8080/index.html || \
	  echo "Open http://localhost:8080/index.html manually" )

# ================================
# Run everything together
# ================================
run:
	make -j2 run-backend run-frontend
	make open

# ================================
# Clean everything
# ================================
clean:
	rm -rf $(VENV)
	find . -type __pycache__ -delete
	@echo "Cleaned build artifacts."
