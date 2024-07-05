---
title: "이진 트리"
tags:
    - Data Structure
date: "2024-07-05"
thumbnail: "https://github.com/mine3873/mine3873.github.io/raw/master/assets/img/thumbnail/book.jpg"
---

# Binary Tree
---
<span style="color:orange">Tree</span>란 노드의 집합에 의한 계층적 자료구조입니다. 각 노드들은 데이터 값과 자식 노드라고 칭하는 노드들의 포인터 값을 가집니다. 그리고 자신의 포인터 값을 가지고 있는 노드를 부모노드라고 합니다.  


트리의 각 노드는 세 가지로 분류될 수 있습니다. 
![1](https://onedrive.live.com/embed?resid=9EB8D569512A5F6B%2132400&authkey=%21AJyLZ72PrUyom9I&width=1327&height=1000)  
1. <span style="color:orange">루트 노드 ( Root Node )</span> : 부모노드를 갖지 않는 경우
2. <span style="color:orange">말단 노드 ( Leaf Node )</span> : 자녀노드를 갖지 않는 경우
3. <span style="color:orange">내부 노드 ( Internal Node )</span> : 위의 두 경우를 제외한 경우

그리고 자신과 같은 부모 노드를 가지고 있는 노드들을 <span style="color:orange">형제 ( Sibling ) 노드</span> 라고 합니다.  

![2](https://onedrive.live.com/embed?resid=9EB8D569512A5F6B%2132401&authkey=%21AGtO9SNff7rdzQs&width=1563&height=1000)  
- 차수 ( Degree, = d ) : 자식 노드의 개수  
- Level ( = Height ) : 루트 노드는 0 또는 1부터 시작하여, 자식노드는 부모노드보다 1을 더한 값을 얻는다.  

### 이진 트리 ( Binary Tree )
<span style="color:orange">이진 트리 ( Binary Tree )</span>는 차수가 2 이하인 트리를 말합니다.  

  
그 안에도 여러 종류가 있습니다. 
### 편향 이진 트리 ( Skewed Binary Tree )
말단 노드를 제외한 모든 노드가 오직 한개의 자식 노드를 갖는 트리를 말합니다.  
Degenerate Tree 라고도 합니다. 이 경우엔 연결리스트와 같은 기능을 가집니다.  
### 정 이진 트리 ( Full Binary Tree )
말단 노드를 제외한 모든 노드의 차수가 0 또는 2인 트리를 말합니다.  
### 완전 이진 트리 ( Complete Binary Tree )
말단 노드를 제외한 모든 노드의 차수가 2인 트리를 말합니다. 또한 마지막 레벨을 제외한 모든 레벨이 채워져 있어야 합니다. 즉, 말단 노드들의 레벨 차이는 최대 1 입니다.  
말단 노드들은 트리에 채워질 때, 왼쪽부터 채워집니다.
### 포화 이진 트리 ( Perfect Binary Tree )
완전 이진 트리에서 마지막 레벨이 모두 채워진 경우를 포화 이진 트리라고 합니다. 즉, 모든 말단 노드의 레벨이 같습니다. 

---

# 이진 탐색 트리 (Binary Search Tree, BST)
이진 탐색 트리는 이진 탐색 알고리즘을 트리에 접목시킨 것입니다.  
각 노드의 데이터보다 작은 값은 왼쪽 부분 트리로, 큰 값은 오른쪽 부분 트리로 저장됩니다.  
특정 값 n을 찾을 때, 현재 노드의 데이터보다 작다면 왼쪽 자식 노드, 크다면 오른쪽 자식 노드로 이동해서 재귀적으로 진행되기 때문에, 이진 탐색과 같이 삽입, 삭제, 탐색 연산이 모두 $O(log_2N)$이 됩니다. 

### 탐색 연산 find "key"
루트 노드부터 탐색을 시작합니다.  
1. 노드의 값 == key 이면, 탐색 성공, 종료
2. 노드의 값 < key 이면, 왼쪽 자식 노드로 이동
3. 노드의 값 > key 이면, 오른쪽 자식 노드로 이동

이 과정을 반복하였을 때, 마지막에 key 값과 같은 값을 가진 노드가 존재하지 않다면 탐색 실패로 종료됩니다.  
### 삽입 연산 insert "key"
삽입 연산이 이루어지기 위해서 탐색 연산을 시작합니다.  
- 탐색이 종료되는 경우, 탐색이 종료된 위치에 키 값을 삽입합니다.  

### 삭제 연산 delete "key"
탐색 연산을 시작합니다.  
- 탐색이 실패하면 어떠한 행동도 취하지 않습니다.  
- 탐색이 성공하면 해당 노드를 삭제합니다.  
    - 해당 노드가 말단 노드인 경우 : 해당 노드를 삭제합니다.  
    - 해당 노드가 하나의 자식 노드를 가지는 경우 : 해당 노드를 삭제하고, 자식 노드로 이를 대체합니다.  
    - 해당 노드가 두개의 자식 노드를 가지는 경우 : 해당 노드를 삭제하고, 다음 중 하나의 노드를 골라 이를 대체합니다.  
        - 왼쪽 부분 트리에서 가장 큰 값을 가지는 노드
        - 오른쪽 부분 트리에서 가장 작은 값을 가지는 노드

이제 BST를 구현해보도록 하겠습니다.  

## code
---
``` cpp
#include<iostream>
#include<ostream>

#define inorder 0
#define preorder 1
#define postorder 2

using namespace std;

class Underflow {};

template <typename T>
class Node {
    public:
    T data;
    Node<T> *left, *right;

    Node<T>() : left(nullptr), right(nullptr) {}
    Node<T>(T value) : data(value), left(nullptr), right(nullptr) {}
};

template <typename T>
class BinaryTree {
    public:
    int size;
    Node<T> *root;
    BinaryTree<T>() : size(0) , root(nullptr) {}
    
    
    void insert(T value){
        insertNode(value, root);
        size++;
    }
    
    bool search(T value){
        return searchNode(value, root);
    }
    void remove(T value){
        try{
            if(size <= 0) throw Underflow();
            removeNode(value, root);
            size--;
            cout << "\nremoved " << value << '\n';
        }
        catch(Underflow e){
            cout << "underflow occurs" << '\n';
            exit(0);
        }
    }

    void print(int N){
        if(N == inorder){
            cout << "inorder : ";
            inOrder(root);
        }
        else if(N == preorder){
            cout << "preorder : ";
            preOrder(root);
        }
        else if(N == postorder){
            cout << "postorder : ";
            postOrder(root);
        }
        cout << '\n';
    }

    private:
    void insertNode(T value, Node<T> *&currentNode){
        if(currentNode == nullptr)
            currentNode = new Node<T>(value);
        else{
            if(value <= currentNode->data)
                insertNode(value,currentNode->left);
            else
                insertNode(value, currentNode->right);
        }
    }
    bool searchNode(T value,Node<T> *current){
        if(current->data == value)
            return 1;
        if(value < current->data)
            searchNode(value, current->left);
        else
            searchNode(value,current->right);
        return 0;
    }

    void getleftMost(Node<T> *&current){
        Node<T> *ptr = current->left, *prev = current;
        while(ptr->right != nullptr){
            prev = ptr;
            ptr = ptr->right;
        }
        prev->right = nullptr;
        current->data = ptr->data;
        delete ptr;
    }

    void removeNode(T value, Node<T> *&current){
        if(current != nullptr){
            if(value < current->data)
                removeNode(value,current->left);
            else if(value > current->data)
                removeNode(value,current->right);
            else if (value == current->data){
                if(current->left == nullptr){
                    Node<T> *ptr = current;
                    current = current->right;
                    delete ptr;
                }
                else if(current->right == nullptr){
                    Node<T> *ptr = current;
                    current = current->left;
                    delete ptr;
                }
                else{
                    getleftMost(current);
                }
            }
        }
    }
    void inOrder(Node<T>* current){
        if(current != nullptr){
            inOrder(current->left);
            cout << "[ " << current->data << " ] ";
            inOrder(current->right);
        }
    }
    void preOrder(Node<T>* current){
        if(current != nullptr){
            cout << "[ " << current->data << " ] ";
            preOrder(current->left);
            preOrder(current->right);
        }
    }
    void postOrder(Node<T>* current){
        if(current != nullptr){
            postOrder(current->left);
            postOrder(current->right);
            cout << "[ " << current->data << " ] ";
        }
    }
};

int main(){
    BinaryTree<int> test;
    int k[] = {3, 9, 4, 1, 10, 5, 9, 10, 7, 2};
    for(int i = 0; i < 10; i++)
        test.insert(k[i]);
    test.print(inorder);
    test.print(preorder);
    test.print(postorder);


    test.remove(3);
    test.print(inorder);
    test.print(preorder);
    test.print(postorder);
    return 0;
}

```

코드 내에서는 빈 트리에 3, 9, 4, 1, 10, 5, 9, 10, 7, 2를 순서대로 삽입을 진행합니다.  
그 과정은 다음과 같습니다.  
![11](https://onedrive.live.com/embed?resid=9EB8D569512A5F6B%2132403&authkey=%21AOudnW6KZa1OmXA&width=1524&height=999)  
![12](https://onedrive.live.com/embed?resid=9EB8D569512A5F6B%2132404&authkey=%21ALUHQgi9zzrl5vc&width=2316&height=1000)  

여기서 inorder, preorder, postorder가 있는데, 이는 이진 트리 순회 방법입니다.  
특정한 순서로 각 노드를 방문하여 최종적으로 트리의 모든 노드를 방문하는 방법을 말합니다.  
- inorder traversal : 왼쪽 자식 노드, 자신, 오른쪽 자식 노드의 순서로 순회하는 방법입니다.  
- preorder traversal : 자신, 왼쪽 자식 노드, 오른쪽 자식 노드의 순서로 순회하는 방법입니다.  
- postorder traversal : 왼쪽 자식 노드, 오른쪽 자식 노드, 자신의 순서로 순회하는 방법입니다.   

## result
---
```
inorder : [ 1 ] [ 2 ] [ 3 ] [ 4 ] [ 5 ] [ 7 ] [ 9 ] [ 9 ] [ 10 ] [ 10 ] 
preorder : [ 3 ] [ 1 ] [ 2 ] [ 9 ] [ 4 ] [ 5 ] [ 9 ] [ 7 ] [ 10 ] [ 10 ]
postorder : [ 2 ] [ 1 ] [ 7 ] [ 9 ] [ 5 ] [ 4 ] [ 10 ] [ 10 ] [ 9 ] [ 3 ]

removed 3
inorder : [ 1 ] [ 2 ] [ 4 ] [ 5 ] [ 7 ] [ 9 ] [ 9 ] [ 10 ] [ 10 ]
preorder : [ 2 ] [ 1 ] [ 9 ] [ 4 ] [ 5 ] [ 9 ] [ 7 ] [ 10 ] [ 10 ]
postorder : [ 1 ] [ 7 ] [ 9 ] [ 5 ] [ 4 ] [ 10 ] [ 10 ] [ 9 ] [ 2 ]
```

예시로 보여드린 트리의 구조가 매우 비효율적입니다.  
이러한 구조의 트리의 경우, 삽입, 삭제, 탐색 연산의 시간 복잡도가 증가하게 됩니다.  
이러한 문제에 대한 해결 방안으로 Height Balanced Tree 중에서,  
AVL Tree와 Red-Black Tree에 대해서 알아보도록 하겠습니다.  
~~물론 내일..~~
