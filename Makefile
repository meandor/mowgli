.PHONY: docker-image
docker-image:
	pipenv lock -r > requirements.txt
	docker build -t mowgli .
