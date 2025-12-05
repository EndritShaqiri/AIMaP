VENV=venv

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

run-backend:
	$(VENV)/bin/uvicorn AI_thrember.api:app --host 0.0.0.0 --port 8000

run-frontend:
	cd web && python3 -m http.server 8080

open:
	sleep 2 && xdg-open http://localhost:8080/index.html || open http://localhost:8080/index.html

run: 
	make -j2 run-backend run-frontend
	make open
