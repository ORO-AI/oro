FROM nginx:alpine

RUN apk add --no-cache gettext nginx-module-njs

COPY docker/proxy/nginx.conf.template /tmp/nginx.conf.template
COPY docker/proxy/validate_model.js /etc/nginx/validate_model.js
COPY docker/proxy/model_pairs.json /etc/nginx/model_pairs.json

CMD ["/bin/sh", "-c", ": ${BACKEND_URL:=https://api.oroagents.com} && : ${BACKEND_HOST:=api.oroagents.com} && : ${OPENROUTER_HOST:=openrouter.ai} && : ${PROXY_RESOLVER:=127.0.0.11 valid=30s} && export BACKEND_URL BACKEND_HOST OPENROUTER_HOST PROXY_RESOLVER && envsubst '${SEARCH_SERVER_URL} ${SEARCH_SERVER_PORT} ${CHUTES_HOST} ${BACKEND_URL} ${BACKEND_HOST} ${OPENROUTER_HOST} ${PROXY_RESOLVER}' < /tmp/nginx.conf.template > /etc/nginx/nginx.conf && exec nginx -g 'daemon off;'"]
