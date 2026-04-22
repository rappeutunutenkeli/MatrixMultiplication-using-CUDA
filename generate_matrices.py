import random
import argparse

def generate_matrix(size, filename):
    with open(filename, 'w') as f:
        for i in range(size):
            row = [str(random.randint(-10, 10)) for _ in range(size)]
            f.write(' '.join(row) + '\n')
    print(f"Матрица {size}x{size} сохранена в {filename}")

def main():
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description='Генератор матриц')
    parser.add_argument('size', type=int, nargs='?', default=200, 
                       help='Размер матрицы (по умолчанию 200)')
    args = parser.parse_args()
    
    # Генерируем две матрицы
    generate_matrix(args.size, 'matrix_a.txt')
    generate_matrix(args.size, 'matrix_b.txt')
    print(f"Сгенерировано 2 матрицы размером {args.size}x{args.size}")

if __name__ == "__main__":
    main()