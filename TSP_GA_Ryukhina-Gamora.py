import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


# параметры алгоритма

# указать:
path = 'D:\\py fi\\TSP_try\\distances_HSE_SPb.csv'  # матрица расстояний без заголовков, симметрична относительно главной диагонали. двойные слеши
start_point = 2  # стартовый город (индексация с нуля)

# менять по желанию:
mutation_percent = 0.05  # вероятность мутации 
mutation_level = 0.075  # уровень мутации (% от количества особей)

survival_percent = 25  # процент выживания популяции
population_size = 200  # размер популяции

max_generations = 600  # максимальное число поколений (итераций) 
max_without_changes = 150 # максимальное число поколений без изменения лучшего организма (маршрута)
max_time = 720  # максимальное время выполнения (в секундах)

do_it_three_times = True  # True, если есть лишнее время. Прогонит три раза и выберет лучший результат
# потому что иногда (редко) оно все-таки немного промахивается и рано заканчивает - это зависит от начальной случайной популяции
real_time_plotting = True #True, если комплюктер позволяет не только считать популяции, но и плоттить их KPI в реальном времени. Для False выведет в конце один график
# XXX: В конце каждого плоттинга выдаёт RuntimeError перед лучшей особью. Я не смог обработать это исключение, но оно ни на что не влияет - даже на do_it_three_times (Константин)

matrix = pd.read_csv(path, index_col = 0)


class Route:
    dist_matrix = matrix
    def __init__(self, lst):
        self.lst = lst

    def dist_calc(self):
        """
        Наша fitting function для экземпляра - списка
        """
        total_distance = float()
        for i in range(len(self.lst) - 1):
            total_distance += self.__class__.dist_matrix.iloc[self.lst[i], self.lst[i + 1]]
        return total_distance

    def crossover(self, other):
        """
        Кроссовер по алгоритму PMX
        """
        # Первое и последнее значение списка мы не трогаем - это нулевая точка
        x, y = pd.Series(self.lst[1:len(self.lst) - 1]), pd.Series(other.lst[1:len(other.lst) - 1])
        initial_parents = [y, x]
        # Нам важны индексы, поэтому используем серии пандаса
        length_of_crossover = int(np.round(0.5 * (len(x))))
        start_of_crossover = int(np.random.choice([i for i in range(len(x) - length_of_crossover)]))
        proto_child = x[:start_of_crossover].append(y[start_of_crossover:start_of_crossover + length_of_crossover]).append(x[start_of_crossover + length_of_crossover:])
        parents = [x[:start_of_crossover].append(x[start_of_crossover + length_of_crossover:]), y[start_of_crossover:start_of_crossover + length_of_crossover]]
        while len(set(proto_child.values)) != len(proto_child):
            for i, parent in enumerate(parents):
                parent_zone = proto_child.index.isin(parent.index)
                for gene in proto_child[parent_zone]:
                    try:
                        duplicate_gene_index = proto_child[~parent_zone][proto_child[~parent_zone] == gene].index[0]
                        proto_child[duplicate_gene_index] = initial_parents[i][proto_child[parent_zone][proto_child[parent_zone] == gene].index[0]]
                    except IndexError:
                        continue
        proto_child = list(proto_child)
        proto_child.insert(0, start_point)
        proto_child.append(start_point)
        return proto_child

    def mutate(self, x):
        # Первое и последнее значение списка мы не трогаем - это нулевая точка                
        # x = pd.Series(self.route_list[1:len(self.route_list) - 1])
        x = pd.Series(x[1:len(x) - 1])

        number_to_mutate = np.ceil(mutation_percent * len(x))
        genes_to_mutate = np.random.choice(len(x), size = int(number_to_mutate), replace=False)
        for i in genes_to_mutate:
            search_field_side = np.ceil(len(x) * mutation_level)
            if i + search_field_side > len(x) - 1:
                search_field_side += search_field_side - (len(x) - i) # Эта строчка может всё попортить при малых размерах популяции < 10
                index_change = int(np.round(np.random.triangular(i - search_field_side, i - 1, len(x))))
                while index_change == i or index_change > len(x) - 1:
                    index_change = int(np.round(np.random.triangular(i - search_field_side, i - 1, len(x))))
                to_change = (x[i], i)
                changer = (x[x.index != i][index_change], x[x == x[x.index != i][index_change]].index)
                x[to_change[1]] = changer[0]
                x[changer[1]] = to_change[0]
            elif i - search_field_side < 0:
                search_field_side += abs(i - search_field_side)
                index_change = int(np.round(np.random.triangular(0, i, i + search_field_side)))
                while index_change == i:
                    index_change = int(np.round(np.random.triangular(0, i, len(x))))
                to_change = (x[i], i)
                changer = (x[x.index != i][index_change], x[x == x[x.index != i][index_change]].index)
                x[to_change[1]] = changer[0]
                x[changer[1]] = to_change[0]
            else:
                index_change = int(np.round(np.random.triangular(i - search_field_side, i, i + search_field_side)))
                while index_change == i:
                    index_change = int(np.round(np.random.triangular(i - search_field_side, i, i + search_field_side)))
                to_change = (x[i], i)
                changer = (x[x.index != i][index_change], x[x == x[x.index != i][index_change]].index) # 
                x[to_change[1]] = changer[0]
                x[changer[1]] = to_change[0]
        mutated = list(x)
        mutated.insert(0, start_point)
        mutated.append(start_point)         
        return mutated


class Population(Route):
    """
    Популяция размера size.
    Объекты-маршруты хранятся в self.routes_obj (как <__main__.Route object at ...).
    Если нужно посмотреть на сам маршрут как список точек - атрибут lst.
    """
    def __init__(self, size):
        Route.__init__(self, None)
        self.size = size
        self.routes_obj = [Route(self.generate_random_route()) for _ in range(size)] 

    def generate_random_route(self):
        """
        Один случайный маршрут с учетом начальной точки, из таких составляется первое поколение.
        """
        self.lst = np.random.choice(range(len(self.dist_matrix.columns)-1), len(self.dist_matrix.columns)-1, replace=False).tolist()
        if start_point in self.lst:
            for n in list(range(len(self.dist_matrix.columns))):
                if n not in self.lst:
                    to_replace = n
            for i,j in enumerate(self.lst):
                if j == start_point:
                    self.lst.pop(i)
                    self.lst.insert(i, to_replace)
            self.lst.append(start_point)
            self.lst.insert(0, start_point)
        else:
            self.lst.append(start_point)
            self.lst.insert(0, start_point)

        return self.lst

    def sum_dist_calc(self):
        """
        Это снова fitting function для каждой особи в популяции + атрибут с суммарным расстоянием
        """
        self.total_dist = [i.dist_calc() for i in self.routes_obj]
        self.pop_dist = sum(self.total_dist)  

        return self.total_dist

    def choose_parents(self):
        """
        Выбор родителей по значению fitting function (расстояние).
        Первый - случайно, второй - наиболее похожий по значению функции.
        """
        distances = self.total_dist
        sorted_by_dist = sorted(list(zip(self.routes_obj, distances)), key=lambda x: x[1]) 

        ind1 = np.random.randint(len(self.routes_obj))

        try:
            ind2 = ind1 + 1
            self.rod1, self.rod2 = sorted_by_dist[ind1][0], sorted_by_dist[ind2][0] 
        except IndexError:
            ind2 = ind1 - 1
            self.rod1, self.rod2 = sorted_by_dist[ind1][0], sorted_by_dist[ind2][0]

        return self.rod1, self.rod2

    def breed(self):
        """
        Делает детей от родителей из предыдущего метода и добавляет их в популяцию как объекты класса.
        """
        child = self.rod1.crossover(self.rod2)  

        try:
            mutated_child = self.mutate(child)  # мутируем ребенка
        except KeyError:  # с вероятностью примерно 1/2000 могут быть необъяснимые проблемы с индексами у pandas, тогда не мутируем
            mutated_child = child
        
        self.mutated_obj = Route(mutated_child)  # инициализируем ребенка в класс
        self.routes_obj.append(self.mutated_obj)  # и добавляем в популяцию
        return self.mutated_obj

    def add_child_dist(self):
        child_dist = float()
        for i in range(len(self.mutated_obj.lst) - 1):
            child_dist += self.dist_matrix.iloc[self.mutated_obj.lst[i], self.mutated_obj.lst[i + 1]]
        
        self.total_dist.append(child_dist)

        return self.total_dist

    def create_pool(self):
        """
        Оставляет только survival_percent % лучших особей по расстоянию.
        """
        inds = []
        crit = np.percentile(self.total_dist, 100-survival_percent)

        for ind, i in enumerate(self.total_dist):
            if i > crit: 
                inds.append(ind)
        for i in sorted(inds, reverse=True): 
            self.routes_obj.pop(i)

        return self.routes_obj

    def renew_population(self):
        """
        Одна итерация: выбор родителей, рождение и мутация ребенка, добавление его в популяцию, убийство неугодных.
        Затем пополнение популяции детьми до исходной численности.
        """
        self.sum_dist_calc()
        self.choose_parents()
        self.breed()
        self.add_child_dist()
        self.create_pool() 

        while len(self.routes_obj) < population_size:
            try:
                self.choose_parents()
                self.breed()
                self.add_child_dist() 
            except KeyError: 
                continue

        return self.routes_obj

    def driver(self):
        """
        Запуск алгоритма. 
        Заканчивает работу, если превышено время выполнения или допустимое число итераций, либо результат перестал улучшаться.
        Два графика динамики изменения - по лучшей особи и по сумме для популяции.
        """
        count = 0
        dist_to_plot = []
        pop_dist_to_plot = []
        control = []

        start = time.time()

        if real_time_plotting == True:
            from matplotlib.animation import FuncAnimation
            fig, (ax, ax1) = plt.subplots(2, 1)
            ln, = ax.plot([], [], lw = 2)
            ax.set_title('Best route distance change')
            ax1.set_title('Sum total population distance change')
            ln1, = ax1.plot([], [], lw = 2, color = 'r')
            line = [ln, ln1]
            self.renew_population()
            def init():
                ax.set_xlim(0, max_generations)
                ax.set_ylim(0, sorted([o.dist_calc() for o in self.routes_obj])[0] * 1.05)
                ax1.set_xlim(0, max_generations)
                ax1.set_ylim(5000, self.pop_dist * 1.05)
                return line
            xdata, ydata, y1data = [], [], []
            def animation(i):
                if i < max_generations:
                    count = i
                    self.renew_population()
                    distances = sorted([o.dist_calc() for o in self.routes_obj])
                    dist_to_plot.append(distances[0])
                    pop_dist_to_plot.append(self.pop_dist)
                    control.append(distances[0])
                    if count > max_without_changes:
                        if (control[count] == [control[count-i] for i in range(1,max_without_changes)]).all():
                            plt.close()
                            return
                    if time.time() - start > max_time:
                        print('Timeout')
                        plt.close()
                        return
                    xdata.append(i)
                    ydata.append(distances[0])
                    y1data.append(self.pop_dist)
                    line[0].set_data(xdata, ydata)
                    line[1].set_data(xdata, y1data)
                    return line
            ani = FuncAnimation(fig, animation, frames=500,
                    init_func=init, blit=True)

            try:
                plt.show()
            except RuntimeError:
                print("it's okay")
                plt.close()
        else:

            while count <= max_generations:
                self.renew_population()
                distances = sorted([i.dist_calc() for i in self.routes_obj])
                dist_to_plot.append(distances[0])
                pop_dist_to_plot.append(self.pop_dist)
                control.append(distances[0])
                if count > max_without_changes:
                    if (control[count] == [control[count-i] for i in range(1,max_without_changes)]).all():  # если лучший не менялся последние max_without_changes поколений
                        break
                if time.time() - start > max_time:
                    print('Timeout')
                    break
                count += 1
            plt.subplot(1,2,1)
            plt.plot(dist_to_plot)
            plt.title('Best route distance change')
            plt.subplot(1,2,2)
            plt.plot(pop_dist_to_plot)
            plt.title('Sum total population distance change')
            plt.show()

        distances = [i.dist_calc() for i in self.routes_obj]
        sorted_by_dist = sorted(list(zip(self.routes_obj, distances)), key=lambda x: x[1]) 

        print(f'Optimal route is {sorted_by_dist[0][0].lst} with a total distance of {sorted_by_dist[0][1]}')
        return sorted_by_dist[0][0].lst, sorted_by_dist[0][1]


if do_it_three_times == True:
    print(min([Population(population_size).driver() for _ in range(3)], key=lambda x: x[1]))
else:
    Population(population_size).driver()


# Гамора Констанин
# Рюхина Полина
# Выражаем благодарность нашим научным руководителям, кошкам и Stack Overflow.
