FROM nginx:alpine

RUN apk add --no-cache gettext sed nginx-module-njs

COPY docker/proxy/nginx.conf.template /tmp/nginx.conf.template
COPY docker/proxy/allowed_models.json /etc/nginx/allowed_models.json
COPY docker/proxy/validate_model.js /etc/nginx/validate_model.js

CMD ["/bin/sh", "-c", "envsubst '${SEARCH_SERVER_URL} ${SEARCH_SERVER_PORT} ${CHUTES_HOST}' < /tmp/nginx.conf.template > /etc/nginx/nginx.conf && exec nginx -g 'daemon off;'"]
