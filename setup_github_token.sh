#!/bin/bash
# Скрипт для налаштування GitHub токена

echo "=== Налаштування GitHub Personal Access Token ==="
echo ""
echo "Крок 1: Створіть токен на GitHub:"
echo "1. Відкрийте https://github.com/settings/tokens"
echo "2. Натисніть 'Generate new token (classic)'"
echo "3. Виберіть права 'repo'"
echo "4. Скопіюйте токен"
echo ""
read -p "Введіть ваш GitHub токен: " TOKEN
read -p "Введіть ваш GitHub username [marmen512]: " USERNAME
USERNAME=${USERNAME:-marmen512}

# Налаштувати credential helper
git config --global credential.helper store

# Додати credentials
echo "https://${USERNAME}:${TOKEN}@github.com" > ~/.git-credentials
chmod 600 ~/.git-credentials

echo ""
echo "✅ Токен налаштовано!"
echo ""
echo "Тепер можна зробити push:"
echo "  cd /media/sony/ext4/github/AI_project"
echo "  git push origin main"
