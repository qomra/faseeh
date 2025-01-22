# create chat-ui/.env.local file if not exists
cp .env.local chat-ui/.env.local

# mkidr chat-ui/src/lib/server/endpoints/local if not exists
mkdir -p chat-ui/src/lib/server/endpoints/local

# copy the backend scripts to chat-ui/src/lib/server/endpoints/local
cp -r endpointLocal.ts chat-ui/src/lib/server/endpoints/local

# copy static/mysam to chat-ui/static/mysam
cp -r static/mysam chat-ui/static/