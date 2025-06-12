#include <iostream>
#include "Ford-Falkerson.cpp"
int main() {
    // Пример использования алгоритма
    int n = 6; // Количество вершин в графе

    // Матрица пропускных способностей (граф)
    std::vector<std::vector<int>> capacity = {
        {0, 16, 13, 0, 0, 0},
        {0, 0, 10, 12, 0, 0},
        {0, 4, 0, 0, 14, 0},
        {0, 0, 9, 0, 0, 20},
        {0, 0, 0, 7, 0, 4},
        {0, 0, 0, 0, 0, 0}
    };

    int source = 0; // Исток
    int sink = 5;   // Сток

    // Вычисление максимального потока
    std::cout << "Максимальный поток: " << fordFalkerson(capacity, source, sink) << std::endl;

    return 0;
}
