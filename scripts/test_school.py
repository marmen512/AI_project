"""
Тестовий скрипт для перевірки системи "Школа"
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Перевірити що всі імпорти працюють"""
    print("[TEST] Перевірка імпортів...")
    
    try:
        from school import GPTTeacher, TRMStudent, SchoolCurriculum, KindergartenLearning
        print("[OK] Всі класи імпортовано")
        return True
    except Exception as e:
        print(f"[ERROR] Помилка імпорту: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_curriculum():
    """Перевірити curriculum"""
    print("\n[TEST] Перевірка curriculum...")
    
    try:
        from school import SchoolCurriculum
        curriculum = SchoolCurriculum()
        print(f"[OK] Curriculum створено: {len(curriculum.topics)} тем")
        
        lesson = curriculum.get_next_lesson()
        if lesson:
            print(f"[OK] Наступний урок: {lesson.topic}")
        else:
            print("[WARN] Немає доступних уроків")
        
        return True
    except Exception as e:
        print(f"[ERROR] Помилка curriculum: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpt_teacher():
    """Перевірити GPT teacher (без завантаження моделі)"""
    print("\n[TEST] Перевірка GPT teacher структури...")
    
    try:
        from school.gpt_teacher import GPTTeacher
        print("[OK] GPTTeacher клас знайдено")
        print(f"[OK] Методи: {[m for m in dir(GPTTeacher) if not m.startswith('_')]}")
        return True
    except Exception as e:
        print(f"[ERROR] Помилка: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("[TEST] Тестування системи 'Школа'")
    print("="*60)
    
    results = []
    results.append(("Імпорти", test_imports()))
    results.append(("Curriculum", test_curriculum()))
    results.append(("GPT Teacher", test_gpt_teacher()))
    
    print("\n" + "="*60)
    print("[REPORT] Результати тестування:")
    print("="*60)
    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {name}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n[OK] Всі тести пройдено!")
    else:
        print("\n[WARN] Деякі тести не пройдено")

