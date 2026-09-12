FROM nginx:alpine

COPY tests/integration/search-server.conf /etc/nginx/conf.d/default.conf
