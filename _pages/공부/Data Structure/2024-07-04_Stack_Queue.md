---
title: "스택, 큐"
tags:
    - Data Structure
date: "2024-07-04"
thumbnail: "https://github.com/mine3873/mine3873.github.io/raw/master/assets/img/thumbnail/book.jpg"
---

# Stack
---
![1](https://onedrive.live.com/embed?resid=9EB8D569512A5F6B%2132395&authkey=%21ALp5Ec-b1g5mN-o&width=2862&height=1000)  
<span style="color:orange">Stack</span>은 FILO(First In Last Out) 형식의, 가장 먼저 추가된 데이터가 가장 나중에 삭제되는 자료구조이다.  
헬스를 하면서 바벨에 원판 끼우고 뺄 때, 가장 나중에 넣은 원판을 가장 먼저 빼는 것, 이것이 스택 자료구조라고 보면 된다.  

스택은 프로세스 메모리 상에서 임시 로컬 데이터를 저장하는데 쓰이기도 한다. 예를 들어 C++ 프로그램을 실행하면 코드 내 함수 파라미터, 지역 변수, 리턴 값 등이 이 스택 자료구조에 저장된다.  

스택에는 push 연산과 pop 연산이 있는데, push 연산은 마지막에 노드를 추가하는 연산이고, pop은 가장 마지막 노드를 제거하는 연산이다.  
스택은 보통 배열 또는 연결리스트를 통해서 구현을 한다.   
## by Linked List
---
### code
---
``` cpp
#include<iostream>
#include<ostream>

#define MAX 10

class Underflow {};

using namespace std;

template <typename T>
class Node {
    public:
    T data;
    Node<T> *next;
    Node<T>() : next(nullptr) {}
    Node<T>(T value, Node<T>* next1) : data(value), next(next1) {}
};

template <typename T>
class Stack {
    public:
    int length;
    Node<T> *top;
    Stack<T>() : length(0), top(nullptr) {}
    bool empty(){ return (length ? false : true); }
    void push(T value){
        top = new Node<T>(value, top);
        length++;
    }
    T pop(){
        try{
            if(length == 0) throw Underflow();
            T returnValue = top->data;
            Node<T> *ptr = top;
            top = top->next;
            delete ptr;
            length--;
            return returnValue;
        }
        catch(Underflow e){
            cout << "Underflow occurs" << '\n';
            exit(0);
        }
    }

    ~Stack<T>(){
        Node<T> *ptr = top, *prev;
        while(ptr != nullptr){
            prev = ptr;
            ptr = ptr->next;
            delete prev;
        }
    }
};

template <typename T>
ostream& operator<< (ostream& scan, Stack<T> &inputArr){
    Node<T> *ptr = inputArr.top;

    while(ptr != nullptr){
        scan << "[ " << ptr->data << " ]-";
        ptr = ptr->next; 
    }
    scan << "[ NULL ]" << '\n';
}

int main(){
    Stack<int> test;
    int k = 4;
    for(int i = 0; i < MAX; i++)
        test.push(k++);
    cout << test;
    test.pop();
    cout << test;
    return 0;
}
```
### result
---
```
[ 13 ]-[ 12 ]-[ 11 ]-[ 10 ]-[ 9 ]-[ 8 ]-[ 7 ]-[ 6 ]-[ 5 ]-[ 4 ]-[ NULL ]
[ 12 ]-[ 11 ]-[ 10 ]-[ 9 ]-[ 8 ]-[ 7 ]-[ 6 ]-[ 5 ]-[ 4 ]-[ NULL ]
```
## by Array
---
### code
---
``` cpp
#include<iostream>
#include<ostream>

#define MAX 10

class Underflow {};

using namespace std;

template <typename T>
class Stack {
    public:
    int length;
    int top;
    T* arr;
    
    Stack<T>() : length(0), top(-1) {}
    Stack<T>(int size) : length(size), top(-1) {
        arr = new T(size);
    }
    ~Stack<T>(){
        delete[] arr;
    }
    bool empty(){ return (length ? false : true); }
    bool full(){ return (top == length - 1 ? true : false); }
    void push(T value){
        if(full()){
            T* newArr = new T[length + MAX];
            for(int i = 0; i < length; i++)
                newArr[i] = arr[i];
            delete[] arr;
            arr = newArr;
        }
        arr[++top] = value;
    }
    T pop(){
        try{
            if(top < 0) throw Underflow();
            return arr[top--];
        }
        catch(Underflow e){
            cout << "underflow occurs" << '\n';
            exit(0);
        }
    }
};

template <typename T>
ostream& operator<< (ostream& scan, Stack<T> &inputArr){
    for(int i = inputArr.top; i >= 0; i--)
        scan << "[ " << inputArr.arr[i]  << " ]-";
    scan << "[ NULL ]" << '\n';
}

int main(){
    Stack<int> test;
    int k = 4;
    for(int i = 0; i < MAX; i++)
        test.push(k++);
    cout << test;
    test.pop();
    cout << test;
    return 0;
}
```
### result
---
```
[ 13 ]-[ 12 ]-[ 11 ]-[ 10 ]-[ 9 ]-[ 8 ]-[ 7 ]-[ 6 ]-[ 5 ]-[ 4 ]-[ NULL ]
[ 12 ]-[ 11 ]-[ 10 ]-[ 9 ]-[ 8 ]-[ 7 ]-[ 6 ]-[ 5 ]-[ 4 ]-[ NULL ]
```

# Queue
---
![2](https://onedrive.live.com/embed?resid=9EB8D569512A5F6B%2132396&authkey=%21APsYkuWeG5F2zLQ&width=2725&height=1000)
<span style="color:orange">Queue</span>는 FIFO (First In First Out) 의 형태로, 가장 먼저 추가된 노드가 가장 먼저 삭제되는 자료구조이다.  
도로 신호등에 걸린 차량들을 생각해보았을 때, 신호등이 초록불이 되면 먼저 도착해 정지했던 차량이 먼저 출발하는데, 큐가 이러한 구조로 되어있다.  
큐는 enqueue와 dequeue 연산이 있는데, enqueue는 가장 마지막에 새로운 노드를 추가하는 연산이고, dequeue는 가장 앞에 있는 (가장 먼저 추가됐었던) 노드를 삭제하는 연산이다.  

이러한 큐 자료구조는 한 예로 실행가능한 상태의 프로세스들이 CPU 자원을 할당받고자 대기 중일 때, 이러한 큐 자료구조 형태로 저장된다.  

## By Linked List
---
### code
---
``` cpp
#include<iostream>
#include<ostream>

#define MAX 10

class Underflow {};

using namespace std;

template <typename T>
class Node {
    public:
    T data;
    Node<T> *next;
    Node<T>() : next(nullptr) {}
    Node<T>(T value, Node<T> *next1) : data(value), next(next1) {}
};

template <typename T>
class Queue {
    public:
    int length;
    Node<T> *front, *rear;

    Queue<T>() : length(0), front(nullptr), rear(nullptr) {}
    ~Queue<T>(){
        Node<T> *ptr = front,*prev;
        while(ptr != nullptr){
            prev = ptr;
            ptr = ptr->next;
            delete prev;
        }
    }
    bool empty() {return (length? false : true);}
    void enqueue(T value){
        if(length == 0)
            front = rear = new Node<T>(value, nullptr);
        else{
            rear->next = new Node<T>(value, nullptr);
            rear = rear->next;
        }
        length++;
    }
    T dequeue(){
        try{
            if(empty()) throw Underflow();
            Node<T> *ptr = front;
            T returnValue = ptr->data;
            front = front->next;
            delete ptr;
            length--;
            return returnValue;
        }
        catch(Underflow e){
            cout << "underflow occurs" << '\n';
            exit(0);
        }
    }
};

template <typename T>
ostream& operator<< (ostream& scan, Queue<T>& inputQueue){
    Node<T> *ptr = inputQueue.front;
    scan << "front-";
    while(ptr !=nullptr){
        scan << "[ " << ptr->data << " ]-";
        ptr = ptr->next;
    }
    scan << "rear" << '\n';
}

int main(){
    Queue<int> test;
    int k = 3;
    for(int i = 0; i < MAX; i++)
        test.enqueue(k++);

    cout << test;
    test.dequeue();
    cout << test;
    return 0;
}
```
### result
---
```
front-[ 3 ]-[ 4 ]-[ 5 ]-[ 6 ]-[ 7 ]-[ 8 ]-[ 9 ]-[ 10 ]-[ 11 ]-[ 12 ]-rear 
front-[ 4 ]-[ 5 ]-[ 6 ]-[ 7 ]-[ 8 ]-[ 9 ]-[ 10 ]-[ 11 ]-[ 12 ]-rear
```
## By Array
---
![3](https://onedrive.live.com/embed?resid=9EB8D569512A5F6B%2132398&authkey=%21AH_tbvKR2wqqnO8&width=1844&height=1000)
큐의 한 형태로 원형 큐 (Circular Queue)가 있는데, 이는 배열을 통해 큐를 구현하였을 때, 배열의 크기는 고정된다는 특징으로 저장 공간이 부족하다는 단점을 해결한 형태의 큐이다.  

- **enqueue**  
    rear 인덱스 값에 1을 더해 업데이트하고, rear 인덱스 값에 데이터를 삽입한다. 만약 rear의 인덱스 값이 배열의 크기 이상이 되면 rear 값을 0으로 초기화한다.   
- **dequeue**  
    배열의 front 인덱스에 해당하는 데이터을 리턴하고, front 인덱스 값에 1을 더해 업데이트 한다. enqueue의 rear와 마찬가지로 front 값이 배열의 크기 이상이 되면 rear 값을 0으로 초기화한다.  
### code
---

``` cpp
#include<iostream>
#include<ostream>

#define MAX 10

using namespace std;

class Underflow {};
class Overflow {};

template <typename T>
class Queue {
    public:
    int length;
    int front,rear;
    T *arr;

    Queue<T>() : length(0), front(-1), rear(-1) {
        arr = new T[MAX];
    }
    ~Queue<T>(){
        delete[] arr;
    }
    bool full(){ return (length == MAX ? 1 : 0); }
    bool empty(){ return (length? 0 : 1); }
    void enqueue(T value){
        try{
            if(full()) throw Overflow();
            rear++;
            if(rear >= MAX) rear %= MAX;
            if(length == 0) front = rear;
            arr[rear] = value;
            length++;
        }
        catch(Overflow e){
            cout << "overflow occurs" << '\n';
            exit(0);
        }
    }
    T dequeue(){
        try{
            if(empty()) throw Underflow();
            T returnValue = arr[front];
            if(front == rear) front = rear = -1;
            else
                front++;
            if(front >= MAX) front %= MAX;
            length--;
            return returnValue;
        }
        catch(Underflow e){
            cout << "underflow occurs" << '\n';
            exit(0);
        }
    }

};

template <typename T>
ostream& operator<<(ostream& scan, Queue<T>& inputQueue){
    scan << "front-";
    int idx = inputQueue.front;
    while(1){
        scan << "[ " << inputQueue.arr[idx] << " ]-";
        if(idx == inputQueue.rear) break;
        idx++;
        if(idx >= MAX) idx %= MAX;
    }
    
    scan << "rear" << '\n';
}

int main(){
    Queue<int> test;
    int k = 3;
    for(int i = 0; i < MAX; i++)
        test.enqueue(k++);
    cout << test;
    test.dequeue();
    cout << test;
    test.enqueue(k);
    cout << test;
    return 0;
}

```

### result
---

```
front-[ 3 ]-[ 4 ]-[ 5 ]-[ 6 ]-[ 7 ]-[ 8 ]-[ 9 ]-[ 10 ]-[ 11 ]-[ 12 ]-rear
front-[ 4 ]-[ 5 ]-[ 6 ]-[ 7 ]-[ 8 ]-[ 9 ]-[ 10 ]-[ 11 ]-[ 12 ]-rear
front-[ 4 ]-[ 5 ]-[ 6 ]-[ 7 ]-[ 8 ]-[ 9 ]-[ 10 ]-[ 11 ]-[ 12 ]-[ 13 ]-rear
```

# Queue By Two Stack

<img src = "https://github.com/mine3873/mine3873.github.io/assets/94094712/98b1ab08-d908-4a5b-9bf1-b99b60456917">  

두개의 스택을 통해서 큐를 구현할 수 있다.  
기본적으로 데이터가 저장되는 스택1과 dequeue 작업을 위해 데이터를 임시적으로 저장할 스택2가 필요한데, enqueue연산의 경우 스택1에 push 연산을 통해 스택의 경우와 똑같이 저장한다. dequeue연산을 위해서는 가장 밑에 있는 데이터를 추출해야 하는데, 스택의 경우, FILO 구조로 가장 위의 있는 데이터가 먼저 추출될 수 밖에 없다. 여기서 임시 저장소 스택2가 사용되는데, 먼저 스택1에 저장된 데이터들을 가장 밑에 있는 데이터가 나올 때까지 pop 연산을 진행한다. 그리고 각 연산마다 추출되는 데이터를 스택2에 그대로 push 연산을 진행한다. 그러면 마지막엔 스택2에는 가장 밑에 있던 데이터를 제외한 나머지 노드들이 삽입된 순서의 반대로 저장되어있을텐데, 밑에 있던 데이터를 추출한 후, 스택2 안에 데이터가 모두 사라질 때까지 pop연산을 진행하고, 연산을 통해 추출된 데이터들을 다시 스택1에 push 연산을 진행한다.  

1. 스택1에서 pop 연산 진행
2. 추출된 데이터를 스택2에 push 연산 진행
3. 1,2 단계를 스택1의 가장 밑에 있는 데이터가 추출될 때까지 진행
4. 목표 데이터 추출
5. 스택2에서 pop연산 진행
6. 추출된 데이터를 스택1에 push연산 진행
7. 5,6 단계를 스택2에 아무 데이터도 남지 않을 때까지 진행  

마찬가지로 스택 또한 두개의 큐를 통해서 구현할 수 있다.  
알고만 있자..