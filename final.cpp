#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <queue>
#include <set>
#include <map>
#include <array>
#include <chrono>
#include <thread>
#include <string>
#include <cstdint>

namespace ansi
{
    constexpr const char *RESET = "\033[0m";
    constexpr const char *WALL = "\033[90m";
    constexpr const char *EMPTY = "\033[2;37m";
    constexpr const char *UNDERLINE = "\033[4m";
    constexpr const char *CLEAR = "\033[2J\033[H";

    constexpr const char *AGENT[] = {
        "\033[1;91m",
        "\033[1;92m",
        "\033[1;93m",
        "\033[1;94m",
        "\033[1;95m",
        "\033[1;96m",
        "\033[1;31m",
        "\033[1;33m",
        "\033[1;35m",
        "\033[1;36m",
    };
    constexpr int N_COLORS = 10;
}
struct Maze
{
    int height, width;
    std::vector<std::vector<bool>> grid;
    Maze(int height, int width)
        : height(height), width(width),
          grid(height, std::vector<bool>(width)) {}
};

struct Coord
{
    int y, x;
    friend bool operator<(const Coord &a, const Coord &b)
    {
        if (a.y != b.y)
            return a.y < b.y;
        return a.x < b.x;
    }
    friend bool operator==(const Coord &a, const Coord &b)
    {
        return a.x == b.x && a.y == b.y;
    }
};

constexpr std::array<Coord, 4> DIRS_4 = {{{0, -1}, {-1, 0}, {0, 1}, {1, 0}}};
constexpr std::array<Coord, 5> DIRS_5 = {{{0, -1}, {-1, 0}, {0, 1}, {1, 0}, {0, 0}}};

struct Vertex
{
    Coord coord;
    int time;
    friend bool operator==(const Vertex &a, const Vertex &b)
    {
        return a.coord == b.coord && a.time == b.time;
    }
    friend bool operator<(const Vertex &a, const Vertex &b)
    {
        if (a.coord < b.coord)
            return true;
        if (b.coord < a.coord)
            return false;
        return a.time < b.time;
    }
};

using Path = std::vector<Coord>;

struct BFSDistance
{
    std::vector<std::vector<int>> distanceToEnd;
    BFSDistance(int height, int width)
        : distanceToEnd(height, std::vector<int>(width, -1)) {}
};

enum ConflictType
{
    VERTEX,
    EDGE
};

struct Constraint
{
    Coord pos, edgeTo;
    ConflictType type;
    int time;
};

struct Agent
{
    std::vector<Constraint> constraints;
    Path path;
};

struct CBSNode
{
    std::vector<Agent> agentPlans;
    int cost;
};

struct Conflict
{
    int agentA, agentB;
    int time;
    ConflictType type;
    Coord pos, edgeTo;
};

struct PlanResult
{
    enum Status
    {
        SUCCESS,
        NO_SOLUTION,
        TIMEOUT,
        INFEASIBLE_ROOT,
        PLACEMENT_FAILED
    };
    Status status = NO_SOLUTION;
    int cost = 0;
    int cbsNodes = 0;
    long long timeMs = 0;
    std::vector<Path> paths;
};

// =============================================================
// Načítání mapy z movingai formátu
// =============================================================
Maze readMazeFromFile(const std::string &fileName)
{
    std::ifstream in(fileName);
    if (!in.is_open())
    {
        std::cerr << "Nelze otevrit mapu: " << fileName << "\n";
        return Maze(0, 0);
    }
    std::string token;
    int height, width;

    in >> token >> token;  // "type octile"
    in >> token >> height; // "height N"
    in >> token >> width;  // "width N"
    in >> token;           // "map"

    Maze maze(height, width);
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            char ch;
            in >> ch;
            maze.grid[r][c] = !(ch == '.' || ch == 'G');
        }
    }
    return maze;
}

size_t maxPathLength(const std::vector<Path> &paths)
{
    size_t m = 0;
    for (auto &p : paths)
        m = std::max(m, p.size());
    return m;
}
size_t maxPathLength(const std::vector<Agent> &agents)
{
    size_t m = 0;
    for (auto &a : agents)
        m = std::max(m, a.path.size());
    return m;
}

// =============================================================
// BFS od cíle (heuristika pro A*)
// =============================================================
BFSDistance runBFS(const Maze &maze, Coord endCoord)
{
    BFSDistance res(maze.height, maze.width);
    std::queue<Vertex> q;

    res.distanceToEnd[endCoord.y][endCoord.x] = 0;
    q.push({endCoord, 0});

    while (!q.empty())
    {
        auto tmp = q.front();
        q.pop();
        Coord curr = tmp.coord;
        int time = tmp.time;

        for (auto d : DIRS_4)
        {
            Coord neigh = {curr.y + d.y, curr.x + d.x};
            if (neigh.x < 0 || neigh.x >= maze.width || neigh.y < 0 || neigh.y >= maze.height)
                continue;
            if (maze.grid[neigh.y][neigh.x])
                continue;
            if (res.distanceToEnd[neigh.y][neigh.x] != -1)
                continue;

            res.distanceToEnd[neigh.y][neigh.x] = time + 1;
            q.push({neigh, time + 1});
        }
    }
    return res;
}

// =============================================================
// Low-level A* nad space-time grafem s constrainty
// =============================================================
Path runAStar(const Maze &maze, const BFSDistance &bfsDistance,
              Coord from, Coord to, const std::vector<Constraint> &constraints)
{
    int hStart = bfsDistance.distanceToEnd[from.y][from.x];
    if (hStart < 0)
        return {};

    auto cmp = [&bfsDistance](const Vertex &a, const Vertex &b)
    {
        return (a.time + bfsDistance.distanceToEnd[a.coord.y][a.coord.x]) >
               (b.time + bfsDistance.distanceToEnd[b.coord.y][b.coord.x]);
    };

    std::priority_queue<Vertex, std::vector<Vertex>, decltype(cmp)> pq(cmp);
    std::map<Vertex, Vertex> parent;

    Vertex dummy = {{INT32_MIN, INT32_MIN}, INT32_MIN};
    pq.push({from, 0});
    parent.insert({{from, 0}, dummy});

    while (!pq.empty())
    {
        Vertex v = pq.top();
        pq.pop();

        if (v.coord == to)
        {
            Path res;
            Vertex curr = v;
            while (!(curr == dummy))
            {
                res.push_back(curr.coord);
                curr = parent[curr];
            }
            std::reverse(res.begin(), res.end());
            return res;
        }

        for (auto d : DIRS_5)
        {
            const Vertex newVertex = {{v.coord.y + d.y, v.coord.x + d.x}, v.time + 1};
            if (newVertex.coord.x < 0 || newVertex.coord.x >= maze.width || newVertex.coord.y < 0 || newVertex.coord.y >= maze.height)
                continue;
            if (maze.grid[newVertex.coord.y][newVertex.coord.x])
                continue;
            if (parent.find(newVertex) != parent.end())
                continue;

            bool blocked = false;
            for (const Constraint &constraint : constraints)
            {
                if (constraint.type == ConflictType::VERTEX && newVertex.coord == constraint.pos && newVertex.time == constraint.time)
                {
                    blocked = true;
                    break;
                }
                if (constraint.type == ConflictType::EDGE && v.coord == constraint.pos && newVertex.coord == constraint.edgeTo && v.time == constraint.time)
                {
                    blocked = true;
                    break;
                }
            }
            if (!blocked)
            {
                pq.push(newVertex);
                parent[newVertex] = v;
            }
        }
    }
    return {};
}

// =============================================================
// Detekce prvního konfliktu (disappearing agents)
// =============================================================
/*Conflict findFirstConflict(const std::vector<Agent> &agentPlans)
{
    for (size_t i = 0; i < agentPlans.size(); i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            size_t shorterLength = std::min(agentPlans[i].path.size(),
                                            agentPlans[j].path.size());
            for (size_t k = 0; k < shorterLength; k++)
            {
                if (agentPlans[i].path[k] == agentPlans[j].path[k])
                    return {(int)i, (int)j, (int)k, ConflictType::VERTEX,
                            agentPlans[i].path[k]};

                if (k + 1 < shorterLength && agentPlans[i].path[k] == agentPlans[j].path[k + 1] && agentPlans[j].path[k] == agentPlans[i].path[k + 1])
                    return {(int)i, (int)j, (int)k, ConflictType::EDGE,
                            agentPlans[i].path[k], agentPlans[i].path[k + 1]};
            }
        }
    }
    return {0, 0, -1};
}*/


Conflict findFirstConflict(const std::vector<Agent> &agentPlans)
{
    size_t longestPath = maxPathLength(agentPlans);
    for(size_t t = 0; t < longestPath; t++)
    {
        for (size_t i = 0; i < agentPlans.size(); i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                if(t >= agentPlans[i].path.size() || t >= agentPlans[j].path.size())continue;
                if (agentPlans[i].path[t] == agentPlans[j].path[t])
                    return {(int)i, (int)j, (int)t, ConflictType::VERTEX,
                            agentPlans[i].path[t]};

                if (t + 1 < agentPlans[i].path.size()&& t + 1 < agentPlans[j].path.size() && agentPlans[i].path[t] == agentPlans[j].path[t + 1] && agentPlans[j].path[t] == agentPlans[i].path[t + 1])
                    return {(int)i, (int)j, (int)t, ConflictType::EDGE,
                            agentPlans[i].path[t], agentPlans[i].path[t + 1]};
            }
        }
    }
    return {0, 0, -1};
}






// =============================================================
// CBS — high-level solver
// =============================================================
PlanResult findPlan(const Maze &maze,
                    const std::vector<std::pair<Coord, Coord>> &agentRoutes,
                    int timeoutMs = 30000)
{
    auto t0 = std::chrono::steady_clock::now();
    PlanResult result;
    auto elapsed = [&]()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - t0)
            .count();
    };

    int nAgents = agentRoutes.size();
    CBSNode root = {std::vector<Agent>(nAgents), 0};
    std::vector<BFSDistance> bfsDistances(nAgents, BFSDistance(maze.height, maze.width));

    for (int agent = 0; agent < nAgents; agent++)
    {
        bfsDistances[agent] = runBFS(maze, agentRoutes[agent].second);
        Coord start = agentRoutes[agent].first;

        if (bfsDistances[agent].distanceToEnd[start.y][start.x] == -1)
        {
            result.status = PlanResult::INFEASIBLE_ROOT;
            result.timeMs = elapsed();
            return result;
        }
        root.agentPlans[agent].path = runAStar(maze, bfsDistances[agent],
                                               start, agentRoutes[agent].second, {});
        if (root.agentPlans[agent].path.empty())
        {
            result.status = PlanResult::INFEASIBLE_ROOT;
            result.timeMs = elapsed();
            return result;
        }
        root.cost += (int)root.agentPlans[agent].path.size();
    }

    auto cmp = [](const CBSNode &a, const CBSNode &b)
    { return a.cost > b.cost; };
    std::priority_queue<CBSNode, std::vector<CBSNode>, decltype(cmp)> pq(cmp);
    pq.push(root);

    while (!pq.empty())
    {
        if (elapsed() > timeoutMs)
        {
            result.status = PlanResult::TIMEOUT;
            result.timeMs = elapsed();
            return result;
        }

        result.cbsNodes++;
        CBSNode popped = pq.top();
        pq.pop();

        Conflict conflict = findFirstConflict(popped.agentPlans);
        if (conflict.time == -1)
        {
            result.status = PlanResult::SUCCESS;
            result.cost = popped.cost;
            for (auto &a : popped.agentPlans)
                result.paths.push_back(a.path);
            result.timeMs = elapsed();
            return result;
        }

        for (int i : {conflict.agentA, conflict.agentB})
        {
            CBSNode newNode = popped;
            Constraint newConstraint = {conflict.pos, conflict.edgeTo,
                                        ConflictType::VERTEX, conflict.time};

            if (conflict.type == ConflictType::EDGE)
            {
                newConstraint.type = ConflictType::EDGE;
                if (i == conflict.agentB)
                {
                    newConstraint.pos = conflict.edgeTo;
                    newConstraint.edgeTo = conflict.pos;
                }
            }
            newNode.agentPlans[i].constraints.push_back(newConstraint);
            newNode.cost -= (int)newNode.agentPlans[i].path.size();
            newNode.agentPlans[i].path = runAStar(maze, bfsDistances[i],
                                                  agentRoutes[i].first, agentRoutes[i].second,
                                                  newNode.agentPlans[i].constraints);
            if (newNode.agentPlans[i].path.empty())
                continue;
            newNode.cost += (int)newNode.agentPlans[i].path.size();
            pq.push(newNode);
        }
    }

    result.status = PlanResult::NO_SOLUTION;
    result.timeMs = elapsed();
    return result;
}

// =============================================================
// Vizualizace
// =============================================================
void printMaze(const Maze &maze)
{
    for (int r = 0; r < maze.height; r++)
    {
        for (int c = 0; c < maze.width; c++)
        {
            if (maze.grid[r][c])
                std::cout << ansi::WALL << '#' << ansi::RESET;
            else
                std::cout << ansi::EMPTY << '.' << ansi::RESET;
        }
        std::cout << '\n';
    }
}

void printState(const Maze &maze, const std::vector<Path> &paths,
                const std::vector<std::pair<Coord, Coord>> &routes, int t)
{
    std::cout << ansi::CLEAR;
    std::cout << "=== t=" << t << " ===\n";

    for (int r = 0; r < maze.height; r++)
    {
        for (int c = 0; c < maze.width; c++)
        {
            int agentHere = -1;
            for (size_t i = 0; i < paths.size(); i++)
            {
                if (t < (int)paths[i].size() &&
                    paths[i][t].y == r && paths[i][t].x == c)
                {
                    agentHere = (int)i;
                    break;
                }
            }
            int goalOf = -1;
            for (size_t i = 0; i < routes.size(); i++)
            {
                if (routes[i].second.y == r && routes[i].second.x == c)
                {
                    goalOf = (int)i;
                    break;
                }
            }

            char id = (agentHere >= 0) ? char('0' + (agentHere % 10))
                      : (goalOf >= 0)  ? char('0' + (goalOf % 10))
                                       : 0;

            if (maze.grid[r][c])
            {
                std::cout << ansi::WALL << '#' << ansi::RESET;
            }
            else if (agentHere >= 0)
            {
                std::cout << ansi::AGENT[agentHere % ansi::N_COLORS]
                          << id << ansi::RESET;
            }
            /*else if (goalOf >= 0)
            {
                std::cout << ansi::AGENT[goalOf % ansi::N_COLORS]
                          << ansi::UNDERLINE << id << ansi::RESET;
            }*/
            else
            {
                std::cout << ansi::EMPTY << '.' << ansi::RESET;
            }
        }
        std::cout << '\n';
    }

    std::cout << "\nLegend: ";
    for (size_t i = 0; i < paths.size(); i++)
    {
        std::cout << ansi::AGENT[i % ansi::N_COLORS]
                  << "agent " << i << ansi::RESET << "  ";
    }
    std::cout << "\n(# = wall, digit = current pos, "
              << ansi::UNDERLINE << "underlined" << ansi::RESET << " = goal)\n";
}



void runAndVisualize(const Maze &maze,
                     const std::vector<std::pair<Coord, Coord>> &routes,
                     int frame_ms = 100, int timeoutMs = 30000)
{
    std::cout << "Map: " << maze.height << "x" << maze.width
              << ", agents: " << routes.size() << "\n\n";
    printMaze(maze);
    std::cout << "\nAgents:\n";
    for (size_t i = 0; i < routes.size(); i++)
    {
        std::cout << "  " << i << ": (" << routes[i].first.y << ","
                  << routes[i].first.x << ") -> (" << routes[i].second.y
                  << "," << routes[i].second.x << ")\n";
    }

    PlanResult result = findPlan(maze, routes, timeoutMs);

    if (result.status != PlanResult::SUCCESS)
    {
        std::cout << "\nFailed: ";
        switch (result.status)
        {
        case PlanResult::INFEASIBLE_ROOT:
            std::cout << "agent unreachable\n";
            break;
        case PlanResult::TIMEOUT:
            std::cout << "timeout (" << result.timeMs << "ms)\n";
            break;
        case PlanResult::NO_SOLUTION:
            std::cout << "no solution\n";
            break;
        default:
            std::cout << "unknown\n";
        }
        return;
    }

    std::cout << "\nSolved in " << result.timeMs << " ms, cost = "
              << result.cost << ", CBS nodes = " << result.cbsNodes << "\n\n";
    for (size_t i = 0; i < result.paths.size(); i++)
    {
        std::cout << "Agent " << i << " (" << result.paths[i].size() << " steps): ";
        for (auto &c : result.paths[i])
            std::cout << "(" << c.y << "," << c.x << ") ";
        std::cout << "\n";
    }

    std::cout << "\nPress Enter to start animation...";
    std::cin.get();

    int maxT = maxPathLength(result.paths);
    for (int t = 0; t < maxT; t++)
    {
        printState(maze, result.paths, routes, t);
        //std::this_thread::sleep_for(std::chrono::milliseconds(frame_ms));
        std::cin.get();
    }
    std::cout << "\n=== Done ===\n";
}

// =============================================================
// Generování náhodného scénáře
// =============================================================
std::vector<std::pair<Coord, Coord>> generateScenario(
    const Maze &maze, int nAgents, int seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> distY(0, maze.height - 1);
    std::uniform_int_distribution<int> distX(0, maze.width - 1);

    std::vector<std::pair<Coord, Coord>> routes;
    std::set<Coord> used;
    constexpr int MAX_TRIES = 10000;

    for (int i = 0; i < nAgents; i++)
    {
        Coord start, goal;
        int tries;

        tries = 0;
        do
        {
            start = {distY(rng), distX(rng)};
            if (++tries > MAX_TRIES)
                return {};
        } while (maze.grid[start.y][start.x] || used.count(start));
        used.insert(start);

        tries = 0;
        do
        {
            goal = {distY(rng), distX(rng)};
            if (++tries > MAX_TRIES)
                return {};
        } while (maze.grid[goal.y][goal.x] || used.count(goal) || goal == start);
        used.insert(goal);

        routes.push_back({start, goal});
    }
    return routes;
}

// =============================================================
// Benchmark — průchod (mapa × n_agentů × seed) → CSV
// =============================================================
void runBenchmark(const std::vector<std::string> &mapFiles,
                  const std::vector<int> &agentCounts,
                  const std::vector<int> &seeds,
                  int timeoutMs,
                  const std::string &outputCsv)
{
    std::ofstream csv(outputCsv);
    csv << "map,n_agents,seed,status,time_ms,cost,cbs_nodes\n";

    for (const auto &mapFile : mapFiles)
    {
        std::cout << "\n=== " << mapFile << " ===\n";
        Maze maze = readMazeFromFile(mapFile);
        if (maze.height == 0)
            continue;

        for (int n : agentCounts)
        {
            int succ = 0, timeout = 0, fail = 0;
            for (int seed : seeds)
            {
                auto routes = generateScenario(maze, n, seed);
                if (routes.empty())
                {
                    csv << mapFile << "," << n << "," << seed
                        << ",placement_failed,0,0,0\n";
                    csv.flush();
                    continue;
                }

                PlanResult r = findPlan(maze, routes, timeoutMs);

                const char *statusStr = "unknown";
                switch (r.status)
                {
                case PlanResult::SUCCESS:
                    statusStr = "success";
                    succ++;
                    break;
                case PlanResult::TIMEOUT:
                    statusStr = "timeout";
                    timeout++;
                    break;
                case PlanResult::NO_SOLUTION:
                    statusStr = "no_solution";
                    fail++;
                    break;
                case PlanResult::INFEASIBLE_ROOT:
                    statusStr = "infeasible";
                    fail++;
                    break;
                default:
                    break;
                }

                csv << mapFile << "," << n << "," << seed << ","
                    << statusStr << "," << r.timeMs << ","
                    << r.cost << "," << r.cbsNodes << "\n";
                csv.flush();
            }
            std::cout << "  n=" << n << ": " << succ << " success, "
                      << timeout << " timeout, " << fail << " fail\n";
        }
    }
    std::cout << "\nResults written to " << outputCsv << "\n";
}

// =============================================================
// Main
// =============================================================
int main(int argc, char **argv)
{
    // --- Benchmark mód ---
    if (argc >= 2 && std::string(argv[1]) == "--benchmark")
    {
        std::vector<std::string> maps = {
            "maps/random-32-32-10.map",
            "maps/room-32-32-4.map",
            "maps/maze-32-32-2.map",
            "maps/random-64-64-10.map",
            "maps/room-64-64-8.map",
            "maps/maze-128-128-10.map",
        };
        std::vector<int> counts = {2, 4, 6, 8, 10, 12, 15, 20, 25, 30};
        std::vector<int> seeds = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        runBenchmark(maps, counts, seeds, 10000, "benchmark.csv");
        return 0;
    }

    // --- Vizualizační mód ---
    if (argc > 1)
    {
        std::cout << "########## Map: " << argv[1] << " ##########\n";
        Maze maze = readMazeFromFile(argv[1]);
        if (maze.height == 0)
            return 1;

        int nAgents = (argc > 2) ? std::atoi(argv[2]) : 5;
        int seed = (argc > 3) ? std::atoi(argv[3]) : 42;

        auto routes = generateScenario(maze, nAgents, seed);
        if (routes.empty())
        {
            std::cerr << "Nelze vygenerovat scenar (mapa moc plna).\n";
            return 1;
        }

        std::cout << "Vygenerovany scenar (seed=" << seed << "):\n";
        for (size_t i = 0; i < routes.size(); i++)
        {
            std::cout << "  agent " << i << ": ("
                      << routes[i].first.y << "," << routes[i].first.x << ") -> ("
                      << routes[i].second.y << "," << routes[i].second.x << ")\n";
        }

        runAndVisualize(maze, routes, 200);
        return 0;
    }

    // --- Bez argumentů: vypiš nápovědu ---
    std::cout << "Usage:\n";
    std::cout << "  " << argv[0] << " <mapfile> [n_agents] [seed]    # vizualizace\n";
    std::cout << "  " << argv[0] << " --benchmark                    # benchmark do benchmark.csv\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << argv[0] << " maps/random-32-32-10.map 8 42\n";
    std::cout << "  " << argv[0] << " maps/maze-128-128-2.map 5\n";
    return 0;
}