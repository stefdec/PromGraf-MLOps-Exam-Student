all: 
	docker-compose up --build -d

stop: 
	docker-compose down

evaluation:
	docker-compose up -d --build evaluation

predict:
	docker-compose up -d --build predict
