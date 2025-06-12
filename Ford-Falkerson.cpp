#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <cstring>

// Функция поиска в ширину (BFS) для проверки достижимости стока из истока.
bool bfs(const std::vector<std::vector<int>> &residualGraph, int source, int sink, std::vector<int> &parent) {
    int n = residualGraph.size();
    std::vector<bool> visited(n, false);
    std::queue<int> q;

    q.push(source);
    visited[source] = true;
    parent[source] = -1;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v = 0; v < n; v++) {
            // Если вершина еще не посещена и остаточная пропускная способность > 0
            if (!visited[v] && residualGraph[u][v] > 0) {
                q.push(v);
                visited[v] = true;
                parent[v] = u;

                // Если мы дошли до стока, путь найден
                if (v == sink)
                    return true;
            }
        }
    }

    return false; // Если путь не найден
}

// Реализация алгоритма Форда-Фалкерсона
int fordFalkerson(std::vector<std::vector<int>> &capacity, int source, int sink) {
    int n = capacity.size();
    std::vector<std::vector<int>> residualGraph = capacity; // Создаем остаточную сеть
    std::vector<int> parent(n); // Массив для хранения пути
    int maxFlow = 0;       // Изначально поток равен 0

    // Пока есть увеличивающий путь
    while (bfs(residualGraph, source, sink, parent)) {
        // Находим минимальную пропускную способность вдоль найденного пути
        int pathFlow = INT_MAX;
        for (int v = sink; v != source; v = parent[v]) {
            int u = parent[v];
            pathFlow = std::min(pathFlow, residualGraph[u][v]);
        }

        // Обновляем остаточную сеть по найденному пути
        for (int v = sink; v != source; v = parent[v]) {
            int u = parent[v];
            residualGraph[u][v] -= pathFlow;
            residualGraph[v][u] += pathFlow;
        }

        // Добавляем поток текущего пути к общему потоку
        maxFlow += pathFlow;
    }

    return maxFlow;
}
