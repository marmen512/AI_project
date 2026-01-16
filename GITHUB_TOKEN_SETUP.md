# Налаштування Personal Access Token для GitHub

## Крок 1: Створення токена на GitHub

1. **Відкрийте GitHub в браузері:**
   - Перейдіть на https://github.com
   - Увійдіть у свій акаунт

2. **Перейдіть до налаштувань токенів:**
   - Натисніть на ваш аватар (правый верхній кут)
   - Виберіть **Settings**
   - У лівому меню виберіть **Developer settings**
   - Виберіть **Personal access tokens** → **Tokens (classic)**

3. **Створіть новий токен:**
   - Натисніть **Generate new token** → **Generate new token (classic)**
   - Введіть **Note** (наприклад: "AI_project_local")
   - Встановіть **Expiration** (рекомендовано: 90 days або No expiration)
   - Виберіть права доступу:
     - ✅ **repo** (повний доступ до репозиторіїв) - ОБОВ'ЯЗКОВО
     - ✅ **workflow** (якщо використовуєте GitHub Actions)
   - Натисніть **Generate token**

4. **Скопіюйте токен:**
   - ⚠️ **ВАЖЛИВО:** Токен показується тільки один раз!
   - Скопіюйте токен і збережіть його в безпечному місці

## Крок 2: Налаштування Git для використання токена

### Варіант A: Використання токена в URL (одноразово)

```bash
cd /media/sony/ext4/github/AI_project
git push https://<ВАШ_ТОКЕН>@github.com/marmen512/AI_project.git main
```

### Варіант B: Збереження токена в credential helper (рекомендовано)

```bash
# Налаштувати credential helper
cd /media/sony/ext4/github/AI_project
git config --global credential.helper store

# Зробити push (git попросить ввести credentials)
git push origin main
# Username: marmen512
# Password: <вставте ваш токен замість пароля>
```

### Варіант C: Використання токена через environment variable

```bash
# Додати в ~/.bashrc або ~/.profile
export GITHUB_TOKEN="ваш_токен_тут"

# Або використати безпосередньо:
GITHUB_TOKEN="ваш_токен" git push https://${GITHUB_TOKEN}@github.com/marmen512/AI_project.git main
```

### Варіант D: Налаштування через git credential

```bash
# Створити файл з credentials
git config --global credential.helper 'store --file ~/.git-credentials'

# Додати credentials (формат: https://username:token@github.com)
echo "https://marmen512:ВАШ_ТОКЕН@github.com" > ~/.git-credentials
chmod 600 ~/.git-credentials

# Тепер можна робити push
git push origin main
```

## Крок 3: Перевірка налаштування

```bash
cd /media/sony/ext4/github/AI_project
git push origin main
```

Якщо все налаштовано правильно, push пройде успішно.

## Безпека

⚠️ **ВАЖЛИВО:**
- Ніколи не комітьте токен в git репозиторій
- Не діліться токеном з іншими
- Якщо токен скомпрометовано, негайно видаліть його на GitHub та створіть новий
- Використовуйте мінімально необхідні права доступу

## Альтернатива: SSH ключі

Якщо ви хочете використовувати SSH замість HTTPS:

1. Створіть SSH ключ:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

2. Додайте публічний ключ на GitHub:
   - Settings → SSH and GPG keys → New SSH key
   - Скопіюйте вміст `~/.ssh/id_ed25519.pub`

3. Налаштуйте git:
```bash
git remote set-url origin git@github.com:marmen512/AI_project.git
git push origin main
```
