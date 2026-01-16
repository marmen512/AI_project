# Інструкція для push до GitHub

Коміт успішно створено:
- Commit hash: 3a93da6
- Повідомлення: "Оновлення репозиторію з AI-27.12.2025-20_00: синхронізація всіх файлів та підготовка до експорту"
- Змінено: 257 файлів

## Варіанти push до GitHub:

### Варіант 1: Використання Personal Access Token (рекомендовано)
```bash
cd /media/sony/ext4/github/AI_project
git push https://<TOKEN>@github.com/marmen512/AI_project.git main
```

### Варіант 2: Налаштування credential helper
```bash
git config --global credential.helper store
git push origin main
# Введіть username та Personal Access Token замість пароля
```

### Варіант 3: SSH ключ (якщо налаштовано)
```bash
git remote set-url origin git@github.com:marmen512/AI_project.git
git push origin main
```

### Варіант 4: Через GitHub CLI
```bash
gh auth login
git push origin main
```

## Створення Personal Access Token:
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token (classic)
3. Виберіть права: repo (повний доступ до репозиторіїв)
4. Скопіюйте токен та використайте його замість пароля

## Поточний статус:
- ✅ Всі зміни додані до git
- ✅ Коміт створено
- ⏳ Очікує push до GitHub
