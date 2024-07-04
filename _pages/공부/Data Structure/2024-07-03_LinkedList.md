---
title: "연결 리스트"
tags:
    - Data Structure
date: "2024-07-03"
thumbnail: "https://github.com/mine3873/mine3873.github.io/raw/master/assets/img/thumbnail/book.jpg"
---
블렌더 모델링 너무 힘들어서 머리 좀 식히고자 복습할 겸 작성합니다..  
<span style="color:red">
</span>
# Linked List
---
![1](https://onedrive.live.com/embed?resid=9EB8D569512A5F6B%2132369&authkey=%21AOqQbwuKTHaFKBA&width=2157&height=1668)
<span style="color:orange">연결리스트</span>는 각 노드가 연속되어 저장되는 배열과 달리, 각 노드들의 위치가 연속되지 않지만, 각자 다음 노드의 위치 정보를 함께 저장하여 연결되는 자료구조이다.  

<span style="color:orange">각 노드는 데이터와 다른 노드에 대한 포인터가 저장된다.</span>  

연결리스트는 노드의 추가, 삭제에 있어서 특정 노드들의 연결만 업데이트하면 되므로 배열보다 월등히 빠르다.  
그러나 특정 노드에 접근하기 위해서는 첫 노드로부터 순차적으로 접근해야하기 때문에 리스트의 노드 수에 따라서 선형적이며, 특정 노드와의 연결이 끊어지게 되는 경우, 끊어진 노드와 연결된 다른 노드들의 정보도 잃어버린다는 단점이 있다.


## Singly Linked List
---
Singly Linked List는 단순히 각 노드가 다음 노드의 포인터만을 가지고 있는 연결 리스트이다.  
이는 운영체제의 파일 시스템에서 연속 할당 (Contiguous Allocation) 방식에서 사용되기도 한다. 

### code
---
``` cpp
#include<iostream>
#include<ostream>

#define MAX 10

using namespace std;

class InvalidIndex {};
class Underflow {};

template <typename T>
class Node{
    public:
    T data;
    Node<T> *nextNode;
    Node<T>() : nextNode(nullptr) {}
    Node<T>(T value, Node<T> *next1) : data(value), nextNode(next1) {};
    ~Node<T>(){
        nextNode = nullptr;
    }
};

template <typename T>
class LinkedList{
    public:
    int length;
    Node<T> *head;

    LinkedList() : length(0), head(nullptr) {}
    ~LinkedList(){
        Node<T> *ptr1 = head, *ptr2;
        while(ptr1 != nullptr){
            ptr2 = ptr1;
            ptr1 = ptr1->nextNode;
            delete ptr2;
        }
    }

    int size(){
        return length;
    }

    void insert(int idx, T value){
        try
        {   
            if(idx < 0 || idx > length) 
                throw InvalidIndex();
            if(idx == 0)
                head = new Node<T>(value, head);
            else{
                Node<T> *ptr = head;
                for(int i = 0; i < idx - 1; i++)
                    ptr = ptr->nextNode;
                ptr->nextNode = new Node<T>(value,ptr->nextNode);
            }
            length++;
        }
        catch(InvalidIndex e){
            cout << "invalid index" << '\n';
            exit(0);
        }
    }
    
    void erase(int idx){
        try{
            if(length <= 0)
                throw Underflow();
            if(idx < 0 || idx >= length) 
                throw InvalidIndex();
            Node<T> *ptr = head;
            if(idx == 0){
                head = head->nextNode;
                delete ptr;
            }
            else{
                for(int i=0; i < idx - 1;i++)
                    ptr = ptr->nextNode;
                Node<T> *deletedNode = ptr->nextNode;
                ptr->nextNode = deletedNode->nextNode;
                delete deletedNode;
            }
            length--;
        }
        catch(InvalidIndex e){
            cout << "invalid index" << '\n';
            exit(0);
        }
        catch(Underflow e){
            cout << "underflow" << '\n';
        }
    }

    bool serach(T value){
        Node<T> *ptr = head;
        while(ptr != nullptr){
            if(ptr->data == value)
                return 1;
            ptr = ptr->nextNode;
        }
        return 0;
    }

    T& operator[] (int idx){
        Node<T> *ptr = head;
        for(int i=0 ; i < idx; i++)
            ptr = ptr->nextNode;
        return ptr->data;
    }
};

template <typename T>
ostream& operator <<(ostream &scan, LinkedList<T> &inputArr){
        Node<T> *ptr = inputArr.head;
        while(ptr != nullptr){
            scan << "[ " << ptr->data << " ] - ";
            ptr = ptr->nextNode;
        }
        scan << "[ NULL ]" << '\n';
}



int main(){
    LinkedList<int> test;
    int k = 4;
    for(int i = 0; i < MAX; i ++)
        test.insert(i,k);
    cout << test;
    return 0;
}
```
### result
---
```
[ 4 ] - [ 5 ] - [ 6 ] - [ 7 ] - [ 8 ] - [ 9 ] - [ 10 ] - [ 11 ] - [ 12 ] - [ 13 ] - [ NULL ]
```

## Doubly Linked List
---
![3](https://onedrive.live.com/embed?resid=9EB8D569512A5F6B%2132374&authkey=%21AN7AaYfg0HtdVIQ&width=2676&height=1000)
Doubly Linked List는 각 노드가 다음 노드의 포인터와 이전 노드의 포인터를 저장하여, 이중으로 연결된 연결리스트를 말한다.  
각 노드가 이중으로 연결되었기 때문에, 혹여나 특정 노드가 다음 노드와의 연결이 끊어지더라도, 끊어진 다음 노드에 저장된 이전 노드 포인터를 이용하여 복원할 수 있다는 장점이 있다.  
### code
---
``` cpp
#include<iostream>
#include<ostream>

#define MAX 10

using namespace std;

class InvalidIndex {};
class Underflow {};

template <typename T>
class Node {
    public:
    T data;
    Node<T> *prev, *next;

    Node<T>() : prev(nullptr), next(nullptr) {}
    Node<T>(T value, Node<T> *prev1, Node<T> *next1) : data(value), prev(prev1), next(next1) {}
};

template <typename T>
class DoublyLinkedList {
    public:
    int length;
    Node<T> *head, *tail;

    DoublyLinkedList<T>() : length(0), head(nullptr), tail(nullptr) {}

    int size() { return length; }
    void insert(int idx, T value){
        try
        {
            if(idx < 0 || idx > length) throw InvalidIndex();

            if(idx == 0) {
                head = new Node<T>(value, nullptr, head);
                if(length == 0)
                    tail = head;
                else
                    head->next->prev = head;
            }
            else{
                tail = new Node<T>(value,tail,nullptr);
                tail->prev->next = tail;
            }
            length++;
        }
        catch(InvalidIndex e)
        {
            cerr << "invalid index" << '\n';
            exit(0);
        }
    }

    void erase(int idx){
        try{
            if(!length) throw Underflow();
            if(idx < 0 || idx >= length) throw InvalidIndex();
            
            if(idx == 0){
                Node<T> *ptr = head;
                head = head->next;
                head->prev = nullptr;
                delete ptr;
            }
            else if(idx == length - 1){
                Node<T> *ptr = tail;
                tail = tail->prev;
                tail->next = nullptr;
                delete ptr;
            }
            else{
                if(idx >= (length - 1) / 2.0){
                    Node<T> *ptr = tail;
                    for(int i = length - 1; i > idx + 1; i--)
                        ptr = ptr->prev;
                    Node<T> *erasedNode = ptr->prev;
                    ptr->prev = erasedNode->prev;
                    ptr->prev->next = ptr;
                    delete erasedNode;

                }
                else{
                    Node<T> *ptr = head;
                    for(int i = 0; i < idx - 1; i++)
                        ptr = ptr->next;
                    Node<T> *eraseNode = ptr->next;
                    ptr->next = eraseNode->next;
                    ptr->next->prev = ptr;
                    delete eraseNode;
                }
            }
            length--;
        }
        catch(Underflow e){
            cout << "underflow occur" << '\n';
            exit(0);
        }
        catch(InvalidIndex e){
            cout << "invalid index" << '\n';
            exit(0);
        }
    }
};

template <typename T>
ostream& operator <<(ostream &scan, DoublyLinkedList<T> &inputArr){
        Node<T> *ptr = inputArr.head;
        scan << "[ NULL ] - ";
        while(ptr != nullptr){
            scan << "[ " << ptr->data << " ] - ";
            ptr = ptr->next;
        }
        scan << "[ NULL ]" << '\n';
}

template <typename T>
ostream& operator >>(ostream &scan, DoublyLinkedList<T> &inputArr){
        Node<T> *ptr = inputArr.tail;
        scan << "[ NULL ] - ";
        while(ptr != nullptr){
            scan << "[ " << ptr->data << " ] - ";
            ptr = ptr->prev;
        }
        scan << "[ NULL ]" << '\n';
}

int main(){
    DoublyLinkedList<int> test;
    int k = 4;
    for(int i = 0; i < MAX; i ++)
        test.insert(i,k++);
    cout << test << '\n';
    cout >> test << '\n';

    test.erase(7);
    cout << "delete indxe.7\n" << test << '\n';
    test.erase(2);
    cout << "delete index.2\n" << test << '\n';
    return 0;
}
```
### result
---
```
[ NULL ] - [ 4 ] - [ 5 ] - [ 6 ] - [ 7 ] - [ 8 ] - [ 9 ] - [ 10 ] - [ 11 ] - [ 12 ] - [ 13 ] - [ NULL ]

[ NULL ] - [ 13 ] - [ 12 ] - [ 11 ] - [ 10 ] - [ 9 ] - [ 8 ] - [ 7 ] - [ 6 ] - [ 5 ] - [ 4 ] - [ NULL ]

delete indxe.7
[ NULL ] - [ 4 ] - [ 5 ] - [ 6 ] - [ 7 ] - [ 8 ] - [ 9 ] - [ 10 ] - [ 12 ] - [ 13 ] - [ NULL ]

delete index.2
[ NULL ] - [ 4 ] - [ 5 ] - [ 7 ] - [ 8 ] - [ 9 ] - [ 10 ] - [ 12 ] - [ 13 ] - [ NULL ]
```
## Circular Linked List
---
![4](https://onedrive.live.com/embed?resid=9EB8D569512A5F6B%2132375&authkey=%21ABaBdEGSM-avqgs&width=2386&height=1491)  
Circular Linked List는 tail, 마지막 노드가 head, 처음 노드와 연결된 구조를 말한다.  
순차적으로 재생되는 음악 재생목록이라고 보면 된다.  
### code
---
``` cpp
##include<iostream>
##include<ostream>

##define MAX 10

using namespace std;

class InvalidIndex {};
class Underflow {};

template <typename T>
class Node {
    public:
    T data;
    Node<T> *prev, *next;

    Node<T>() : prev(nullptr), next(nullptr) {}
    Node<T>(T value, Node<T> *prev1, Node<T> *next1) : data(value), prev(prev1), next(next1) {}
};

template <typename T>
class CircularLinkedList {
    public:
    int length;
    Node<T> *head, *tail;

    CircularLinkedList<T>() : length(0), head(nullptr), tail(nullptr) {}
    
    int size(){ return length; }

    void insert(int idx, T value){
        try{
            if(idx < 0 || idx > length) throw InvalidIndex();
            if(idx == 0){
                head = new Node<T>(value, tail, head);
                if(length == 0)
                    tail = head;
                else
                    head->next->prev = head;
                
            }
            else{
                tail = new Node<T>(value,tail,head);
                tail->prev->next = tail;
                tail->next->prev = tail;
            }
            length++;
        }   
        catch(InvalidIndex e){
            cout << "invalid index" << '\n';
            exit(0);
        }
    }
    void erase(int idx){
        try{
            if(!length) throw Underflow();
            if(idx < 0 || idx >= length) throw InvalidIndex();

            if(idx == 0){
                Node<T> *ptr = head;
                head = head->next;
                head->prev = tail;
                tail->next = head;
                delete ptr;
            }
            else if(idx == length - 1){
                Node<T> *ptr = tail;
                tail = tail->prev;
                tail->next = head;
                head->prev = tail;
                delete ptr;
            }
            else{
                if(idx >= (length - 1) / 2.0){
                    Node<T> *ptr = tail;
                    for(int i = length - 1; i > idx + 1; i--)
                        ptr = ptr->prev;
                    Node<T> *erasedNode = ptr->prev;
                    ptr->prev = erasedNode->prev;
                    ptr->prev->next = ptr;
                    delete erasedNode;
                }
                else{
                    Node<T> *ptr = head;
                    for(int i = 0; i < idx - 1; i++)
                        ptr = ptr->next;
                    Node<T> *erasedNode = ptr->next;
                    ptr->next = erasedNode->next;
                    ptr->next->prev = ptr;  
                    delete erasedNode;
                }
            }
            length--;
        }
        catch(Underflow e){
            cout << "underflow occur" << '\n';
            exit(0);
        }
        catch(InvalidIndex e){
            cout << "invalid index" << '\n';
            exit(0);
        }
    }
};

template <typename T>
ostream& operator <<(ostream &scan, CircularLinkedList<T> &inputArr){
    Node<T> *ptr = inputArr.head;
    for(int i = 0; i < inputArr.size(); i++){
        scan << "[ " << ptr->data << " ] - ";
        ptr = ptr->next;
    }
    scan << "[ " << ptr->data << " ]" << '\n';
}

template <typename T>
ostream& operator >>(ostream &scan, CircularLinkedList<T> &inputArr){
    Node<T> *ptr = inputArr.tail;
    for(int i = inputArr.size() - 1; i >= 0; i--){
        scan << "[ " << ptr->data << " ] - ";
        ptr = ptr->prev;
    }
    scan << "[ " << ptr->data << " ]" << '\n';
}

int main(){
    CircularLinkedList<int> test;
    int k = 4;
    for(int i = 0; i < MAX; i ++)
        test.insert(i,k++);
    cout << test << '\n';
    cout >> test << '\n';

    test.erase(7);
    cout << "delete indxe.7\n" << test << '\n';
    test.erase(2);
    cout << "delete index.2\n" << test << '\n';

    return 0;
}
```
### result
---
```
[ 4 ] - [ 5 ] - [ 6 ] - [ 7 ] - [ 8 ] - [ 9 ] - [ 10 ] - [ 11 ] - [ 12 ] - [ 13 ] - [ 4 ]

[ 13 ] - [ 12 ] - [ 11 ] - [ 10 ] - [ 9 ] - [ 8 ] - [ 7 ] - [ 6 ] - [ 5 ] - [ 4 ] - [ 13 ]

delete indxe.7
[ 4 ] - [ 5 ] - [ 6 ] - [ 7 ] - [ 8 ] - [ 9 ] - [ 10 ] - [ 12 ] - [ 13 ] - [ 4 ]

delete index.2
[ 4 ] - [ 5 ] - [ 7 ] - [ 8 ] - [ 9 ] - [ 10 ] - [ 12 ] - [ 13 ] - [ 4 ]
```
