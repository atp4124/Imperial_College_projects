"""Scientific Computation Project 2, part 1
Your CID here:01365348
"""

from collections import deque

def flightLegs(Alist,start,dest):
    """
    Question 1.1
    Find the minimum number of flights required to travel between start and dest,
    and  determine the number of distinct routes between start and dest which
    require this minimum number of flights.
    Input:
        Alist: Adjacency list for airport network
        start, dest: starting and destination airports for journey.
        Airports are numbered from 0 to N-1 (inclusive) where N = len(Alist)
    Output:
        Flights: 2-element list containing:
        [the min number of flights between start and dest, the number of distinct
        jouneys with this min number]
        Return an empty list if no journey exist which connect start and dest
    """

    
    #keep track of explored nodes
    explored=[]
    #keep track of all the paths to be checked
    queue=deque()
    queue.append([start])
    list_flights=[[],[]]
    dmin=float("inf")
    #return path if start is goal
    if start==dest:   
        print ('Start==destination')
    #keep looping until all possible paths have been checked
    while queue:
        #pop the first  path from the queue
        path=queue.popleft()
        #get the last  node from the path
        node=path[-1]
        if node  not in explored:
            neighbours=Alist[node]
            
            #go through all the neighbour nodes, construct a new path and
            #push it into the queue
            for neighbour in neighbours:
                new_path=path.copy()
                new_path.append(neighbour)
                queue.append(new_path)
                #return path if neighbour is goal
                m=len(new_path)
                if neighbour==dest and m<=dmin :
                    list_flights[0].append(new_path)
                    list_flights[1].append(m)
                dmin=m
                    
            #mark node as explored
            explored.append(node)
    if not list_flights:
        return []
    else:
        minimum=list_flights[1][0]
        indices=len(list_flights[0])
        return [minimum-1,indices]

def deepcopy(A):
    rt = []
    for elem in A:
        if isinstance(elem,list):
            rt.append(deepcopy(elem))
        else:
            rt.append(elem)
    return rt

def safeJourney(Alist,start,dest):
    """
    Question 1.2 i)
    Find safest journey from station start to dest
    Input:
        Alist: List whose ith element contains list of 2-element tuples. The first element
        of the tuple is a station with a direct connection to station i, and the second element is
        the density for the connection.
    start, dest: starting and destination stations for journey.

    Output:
        Slist: Two element list containing safest journey and safety factor for safest journey
    """
    explored=[]
    queue=deque()
    queue.append([[start],0])
    list_stations=[]
    dmin=float("inf")
    if start==dest:
        print("Start=destination")
    while queue:
        path=queue.popleft()
        node=path[0][-1]
        if node not in explored:
            neighbours=Alist[node]
         
            for neighbour,density in neighbours:
                new_path=deepcopy(path)
                new_path[0].append(neighbour)
                if new_path[1]<density:
                    new_path[1]=density
                queue.append(new_path)
                if neighbour==dest and new_path[1]<=dmin:
                    list_stations.append(new_path)
                dmin=new_path[1]
            explored.append(node)
    if not list_stations:
        return []
    else:
        return list_stations[0]
    
    

def shortJourney(Alist, start, dest):
        """
    Question 1.2 ii)
    Find shortest journey from station start to dest. If multiple shortest journeys
    exist, select journey which goes through the smallest number of stations.
    Input:
        Alist: List whose ith element contains list of 2-element tuples. The first element
        of the tuple is a station with a direct connection to station i, and the second element is
        the time for the connection (rounded to the nearest minute).
    start, dest: starting and destination stations for journey.

    Output:
        Slist: Two element list containing shortest journey and duration of shortest journey
    """
 
        Slist=[[],None]
        inf=float("inf")
        vertices=set(range(len(Alist)))
        dist = {vertex: inf for vertex in vertices}
        previous = {vertex: None for vertex in vertices}
        dist[start] = 0
        q = vertices.copy()
        while q:
            u = min(q, key=lambda vertex: dist[vertex])
            q.remove(u)
            if dist[u] == inf or u == dest:
                break
            for v, cost in Alist[u]:
                alt = dist[u] + cost
                if alt < dist[v]:                                  # Relax (u,v,a)
                    dist[v] = alt
                    previous[v] = u
            
        s, u = deque(), dest
        while u!=start:
            s.appendleft(u)
            u=previous[u]
        s.appendleft(u)
        
        Slist[0]=list(s)
        Slist[1]=dist[dest]
        
        return Slist
   
def there_is_path(Alist,start,dest): 
   
    #keep track of explored nodes
    explored=[]
    #keep track of all the paths to be checked
    queue=deque()
    queue.append([start])
   
   
   #return path if start is goal
    if start==dest:   
        print ('Start==destination')
    #keep looping until all possible paths have been checked
    while queue:
        #pop the first  path from the queue
        path=queue.popleft()
       
        #get the last  node from the path
        node=path[-1]
        if node  not in explored:
            neighbours=Alist[node]
            
            #go through all the neighbour nodes, construct a new path and
            #push it into the queue
            for neighbour in neighbours:
               
                new_path=list(path)
               
                new_path.append(neighbour)
                
                queue.append(new_path)
              
                #return path if neighbour is goal
                if neighbour==dest:
                   return new_path
            explored.append(node)
    return []
               
    
        
def cheapCycling(SList,CList):
    """
    Question 1.3
    Find first and last stations for cheapest cycling trip
    Input:
        Slist: list whose ith element contains cheapest fare for arrival at and
        return from the ith station (stored in a 2-element list or tuple)
        Clist: list whose ith element contains a list of stations which can be
        cycled to directly from station i
    Stations are numbered from 0 to N-1 with N = len(Slist) = len(Clist)
    Output:
        stations: two-element list containing first and last stations of journey
    """
    
  
    a=len(SList)
    vertices=range(a)
    comb=[]
    for i in vertices:
        first_fare=SList[i][0]
        for j in vertices:
            if i==j:
                pass
            else:
               fare=SList[j][1]
               list_nodes=(i,j)
               final_fare=first_fare+fare
               comb.append((list_nodes,final_fare))
    sorted_comb=sorted(comb,key=lambda x:x[1])
    for (x,y),z in sorted_comb:
        if there_is_path(CList,x,y):
            return [x,y]
        
    

if __name__=='__main__':
    #add code here if/as desired
    L=None #modify as needed
