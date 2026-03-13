from modules.utilities.logging import log

def main():
    test_payload = {'test': 'testing', 'assessment': 'quiz', 'exam': 'midterm'}
    log(test_payload, 'tests')

if __name__ == '__main__':
    main()
