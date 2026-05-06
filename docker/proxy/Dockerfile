FROM nginx:alpine

RUN apk add --no-cache gettext nginx-module-njs

COPY docker/proxy/nginx.conf.template /tmp/nginx.conf.template
COPY docker/proxy/validate_model.js /etc/nginx/validate_model.js

CMD ["/bin/sh", "-c", ": ${BACKEND_URL:=https://api.oroagents.com} && : ${BACKEND_HOST:=api.oroagents.com} && : ${OPENROUTER_HOST:=openrouter.ai} && export BACKEND_URL BACKEND_HOST OPENROUTER_HOST && envsubst '${SEARCH_SERVER_URL} ${SEARCH_SERVER_PORT} ${CHUTES_HOST} ${BACKEND_URL} ${BACKEND_HOST} ${OPENROUTER_HOST}' < /tmp/nginx.conf.template > /etc/nginx/nginx.conf && exec nginx -g 'daemon off;'"]
