.PHONY: all stop evaluation predict fire-alert

all:
	docker-compose up --build -d

stop: 
	docker-compose down

evaluation:
	docker-compose up -d --build evaluation

predict:
	docker-compose up -d --build predict

fire-alert:
	curl --fail --silent --show-error http://localhost:8080/false_rmse
	@printf '\n'
