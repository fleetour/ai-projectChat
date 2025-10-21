#!/bin/bash
# set_mistral_key.sh — save your Mistral API key locally

read -p "Enter your Mistral API key: " MISTRAL_KEY

# export for current session
export MISTRAL_API_KEY="$MISTRAL_KEY"

# persist in .env for next runs
echo "export MISTRAL_API_KEY=\"$MISTRAL_KEY\"" > .env

echo "✅ API key saved. Remember to run 'source .env' before starting the server next time."
