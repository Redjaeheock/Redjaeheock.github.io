## 연산자 오버로딩2

### < 반드시 해야 하는 대입 연산자의 오버로딩 >

연산자 오버로딩의 필요성에 대한 내용을 다루기 전에 **복사 생성자**에 대해서 다시 다뤄본다면

__복사 생성자__
- 정의하지 않으면 **Default 복사 생성자**가 삽입됨(컴파일러)
- Default 복사 생성자는 멤버 대 멤버의 복사(얕은복사)를 진행함
- 생성자 내에서 동적 할당하거나 깊은 복사가 필요할 경우 작성자가 직접 정의 해야함


다음은 **대입 연산자** 의 대표적인 특성이다


__대입 연산자__
- 정의하지 않으면 **Default 대입 연산자**가 삽입됨(컴파일러)
- Default대입 연산자는 멤버 대 멤버의 복사(얕은 복사)를 진행함
- 연산자 내에서 동적 할당하거나 깊은 복사가 필요할 경우 작성자가 직접 정의 해야함


이렇게 유사한 둘의 차이점은 호출 시점이 된다.
```c++
int main()
{
	Point pos1(5, 7);
	Point pos2 = pos1; // '초기화 과정'이기 때문에 대입 연산자가 아닌 복사 생성자 호출
	...
}
```
이 경우는 복사 생성자의 호출인 상황으로써

**pos2 를  '초기화 하는 과정' 에서 이미 생성된 객체인 pos1**을 사용한 것이다.


```c++
int main()
{
	Point pos1(5, 7);
	Point pos2(3, 9);
	pos2 = pos1; // '대입' 과정이기 때문에 대입 연산자 호출
	...
}
```
이 경우는 대입 연산자의 호출인 상황으로써

**이미 생성된 객체 pos2 를 이미 생성된 객체 pos1 로 '대입'**을 하는 것이다.

---
## Default 대입 연산자의 문제점

```c++
class Person
{
private:
	char *name;
	int age;
public:
	Person(char *myname, int mayage)
	{
		int len = strlen(nyname) + 1;
		name = new char[len]; // 동적 할당 주시
		strcpy(name, myname);
		age = myage;
	}
	void ShowPersonInfo() const
	{
		cout << "이름: " << name << endl;
		cout << "나이: " << age << endl;
	}
	~Person()
	{
		delete []name;
		cout << "called destructor!" << endl;
	}
};
```

```c++
int main()
{
	Person man1("Lee dong woo", 29);
	Person man2("You ji yul", 22);
	
	man2 = man1;
	man1.ShowPersonInfo();
	man2.ShowPersonInfo();
	
	return 0;
}
----------------------------------------------------------------------------
```
```
이름: Lee dong woo
나이: 29
이름: LEe dong woo
나이: 29
called destructor!
```

출력 결과를 보면 'called destructor'가  한 번만 출력된 것을 확인할 수 있다
(물론 컴파일러 마다 둘 다 출력하거나 하나만 출력하는 케이스가 있겠지만 본문은 조금더 가시적으로 돕기 위해 둘 다 출력해야 되는 것을 기대값으로 설명해본다)

출력이 하나가 됐든 두 개가 되었든 **객체의 소멸 과정**에서 문제가 발생했단 것을 짐작 할 수 있다.

**operator+()** 를 정의하지 않았기 때문에, **Default 대입 연산자**가 삽입되면서
**Person 생성자** 내부에서 name 에 **동적 할당**된 메모리가 **얕은복사**가 된 채로 man1 과 man2 가
같은 주소를 공유하고 있는 상태인데

여기서 **man1이 먼저 소멸**되면서 **name 의 메모리가 해제** 되어, 

이후 **같은 주소를 소유하던 man2** 의 소멸자 입장에선 해제할 메모리가 없어 
**Double free**가 유도되는 상황이 되버린다

이는 다른 챕터에서 복사 생성자의 **얕은 복사에서의 문제점**과 동일한 내용임을 알 수 있다.

이를 방지하기 위해서 **깊은 복사가 필요**하고 그로 인해 **반드시 대입 연산자의 오버로딩**의 정의가 필요한 것이다.

```c++
Person& operator=(const Person& ref)
{
	delete []name; // 기존에 소유하고 있을 메모리 해제
	
	int len = strlen(nyname) + 1;
	name = new char[len]; // 동적 할당 주시
	strcpy(name, ref.name);
	age = ref.age;
	
	return *this;
}
```

---
## 상속 구조에서의 대입 연산자 호출

```c++
class First
{
private:
	int num1, num2;
public:
	First(int n1=0, int n2=0) : num1(n1), num2(n2)
	{}
	void ShowData()
	{
		cout << num1 << ", " << num2 << endl; 
	}
	First& operator=(const First& ref)
	{
		cout << "First& operator=()" << endl;
		num1 = ref.num1;
		num2 = ref.num2;
		
		return *this;
	}
};
```

```c++
class Second : public First
{
private:
	int num3, num4;
public:
	Second(int n1, int n2, int n3, int n4)
		: First(n1, n2), num3(n3), num4(n4)
	{}
	void ShowData()
	{
		First::ShowData();
		cout << num3 << ", " << num4 << endl;
	}
	/*
	Second& operator=(const Second& ref)
	{
		cout << "Second& operator=()" << endl;
		num1 = ref.num3;
		num2 = ref.num4;
		
		return *this;
	}
	*/
};
```

```c++
int main()
{
	Second ssrc(111, 222, 333, 444);
	Second scpy(0, 0, 0, 0);
	scpy = ssrc;
	scpy.ShowData();
	
	return 0;
}
---------------------------------------------------------------------------------
```
```
First& operator=()
111, 222
333, 444
```

출력 결과를 통해서 알 수 있는 것은

**Second 객체의 대입 연산자를 주석** 처리한 후 객체끼리의 대입을 실행했을 때,

Second 클래스에선 대입 연산자가 없으니 **컴파일러가 삽입한 Default 대입 연산자**가 실행이 되고, 
이 Default 대입 연산자가 **First 클래스의 대입 연산자**까지 호출한다는 것을 유추할 수 있는데,

**First::operator=() 의 출력문**이 출력되어 있고, 
**scpy.ShowData()** 의 결과로 **scpy 의 상위 영역의 멤버 값 (0, 0) 이 (111, 222) 로 출력**된 
부분이 유추를 가능케 하는 대목이다


하지만 우린 **대입 연산자를 직접 정의하는 것을 지향**하기 때문에 **scpy 의 주석을 해제**하고 실행한다면 다음과 같은 결과를 얻을 수 있다.

```c++
---------------------------------------------------------------------------------
```
```
Second& operator=()
0, 0
333, 444
```
직접 정의한 코드에선 **First 클래스 영역 멤버 변수**에 대한 내용을 다루지 않았기 때문에,
자연스럽게 **Second 클래스 영역 멤버 변수만 대입**처리가 되어 있는 것을 볼 수 있다.

우린 그럼 c++ 의 자연스러운 동작을 유지하기 위해 **First 클래스 영역 멤버 변수의 대입**을 유도하는 것 또한 책임을 지어야 한다.

그렇다고 **First 클래스의 멤버를 직접 접근해서 대입**하는 것은 c++ 의 원칙에 맞지 않으니, 
아래의 코드 처럼 **First 클래스의 operator=() 를 호출**하는 것이 바람직 한 것으로 사료된다.

```c++
Second& operator=(const Second& ref)
{
	cout << "Second& operator=()" << endl;
	First::operator=(ref); // 상위 클래스의 대입 연산자 호출 정의로 책임을 다함
	num1 = ref.num3;
	num2 = ref.num4;
	
	return *this;
}
```


여기서 중요한 점은 **멤버의 동적 할당이나 특별히 적접 정의할 내용**이 없다면, 
**상속과 피상속 클래스에서 대입 연산자의 직접 정의 없이 컴파일러의 Default 대입 연산자**에게 
맡기는 게 작성자의 실수보다 조금 더 안전한 동작을 기대할 수 있다는 것이다.

---
## 이니셜라디저의 성능 향상 도움

```c++
class AAA
{
private:
	int num;
public:
	AAA(int n=0) : num(n)
	{
		cout << "AAA(int n=0)" << endl;
	}
	AAA(const AAA& ref) : num(ref.num)
	{
		cout << "AAA(const AAA& ref)" << endl;		
	}
	AAA& operator=(const AAA& ref)
	{
		num=ref.num;
		cout << "operator=(cosnt AAA& ref)" << endl;
		
		return *this;
	}
};
```

```c++
class BBB
{
private:
	AAA mem; // BBB 클래스 내부에서 AAA 클래스를 멤버로 생성
public:
	BBB(const AAA& ref) : mem(ref)
	{}
}
```

```c++
class CCC
{
private:
	AAA mem; // CCC 클래스 내부에서 AAA 클래스를 멤버로 생성
public:
	BBB(const AAA& ref)
	{
		mem=ref;
	}
}
```
```c++
int main()
{
	AAA obj1(12);
	cout << "********************" << endl;
	BBB obj2(obj1);
	cout << "********************" << endl;
	CCC obj3(obj1);

	return 0;
}
---------------------------------------------------------------------------------
```
```
AAA(int n=0)
********************
AAA(const AAA& ref)
********************
AAA(int n=0)
operator=(const AAA& ref)
```

결과에서 알 수 있듯이 **이니셜라이저**를 통한 값은 **객체 생성과 동시에** 한 번에 정리되는 한 편,
**대입연산**을 통한 값은 **객체 생성 후 대입 연산자를 별도 호출**함으로써 두 번의 동작이 이뤄지는 것을 볼 수 있다.

이처럼 객체의 초기화 과정에선 가능한 **이니셜라이저를 사용할 것을 지향**하는 바이다.

---
## 배열의 인덱스 연산자 오버로딩

\[] 연산자를 오버로딩 하면, 배열보다 나은 배열 클래스를 사용할 수 있다.

우선 C, C++ 에서의 배열은 **경계 검사**를 하지 않는다.

그렇기 때문에 배열의 **범위를 넘어선 영역을 인덱싱** 하더라도 쓰레기 값을 읽어 오던가, 
프로그램이 터질 뿐 **컴파일러가 에러 메시지**를 반환하진 않는다.

이러한 특성이 이점으로의 작용도 있겠지만, 여기선 단점으로의 시선으로 다가가고자 한다.

이런 단점을 클래스를 통해 배열을 만든다면 **조금 더 안전한 배열 접근을 유도** 할 수 있다.


우선 **\[] 연산자 함수** 역시 **`operator[]()`** 로 작성하며, 호출시 인자 전달 메커니즘 또한 여느 연산자들과 다를 바가 없다.

`arrObject[2]`

라고 할 경우, arrObject 의 operator\[]() 를 호출하고, operator\[])() 함수의 구조는 
**'operator\[](int idx) { ... }** 이렇게 되어 있을꺼라 유추할 수 있고

호출시 **'arrObject.operator\[](2);'** 와 같다는 것도 알 수 있다.

이러한 특징을 반영해서 예시 코드를 본다면
```c++
class BoundCheckIntArray
{
private:
	int *arr;
	int arrlen;
public:
	BoundCheckIntArray(int len) : arrlen(len)
	{
		arr = new int[len];
	}
	int& operator[](int idx)
	{
		if(idx < 0 || idx >= arrlen)
		{
			cout << "Array index out of bound exception" << endl;
			exit(1);
		}
		
		return arr[idx];
	}
	~BoundCheckIntArray
	{
		delete []arr;
	}
};
```
```c++
int main()
{
	BoundCheckIntArray arr(5)
	for(int i = 0; i < 5; i++)
		arr[i] = (i + 1) * 11;
	for(int i = 0; i < 6; i++)
		cout << arr[i] << endl;
		
	return 0;
}
---------------------------------------------------------------------------------
```
```
11
22
33
44
55
Array index out of bound exception
```
이렇게 출력 결과에서 볼 수 있듯이 잘못된 배열 접근의 확인이 가능하게 되어 배열 졉근의 안전성을 
보장 받을 수 있다.

### \[ 배열 클래스의 안전성 확보 ]

배열은 저장소의 일종이고, 저장소에 저장된 데이터는 **유일성이 보장**되어야 하기 때문에
상솽에 따라 배열 객체를 대상으로 하는 **복사와 관련된 연산은 모두 불가능하게** 해야할 필요성도 있다.

```c
int arr1[3] = {1, 2, 3};
int arr2[3] = {0, 0, 0};

arr1 = arr2; // 허용되지 않는 구문
```
이런식의 배열 자체의 대입 연산 또한 방지하고 싶다면 아래 예시 코드처럼

**복사 생성자와 대입 연산자를 private 에서 선언**하면 위와 같은 기능을 추가할 수 있다.

```c++
class BoundCheckIntArray
{
private:
	int *arr;
	int arrlen;
	BoundCheckIntArray(const BoundCheckIntArray& arr) {}
	BoundCheckIntArray& operator=(const BoundCheckIntArray& arr) {}
public:
	...
}
```
### \[ const 함수를 이용한 오버로딩의 활용 ]

함수의 **const 유무**는 **오버로딩의 조건** 내용이기 때문에

```c++
int operator[](int idx) const // int 반환, const설정
{
	if(idx < 0 || idx >= arrlen)
	{
		cout << "Array index out of bound exception" << endl;
		exit(1);
	}
	
	return arr[idx];
}
```
```c++
int& operator[](int idx) //int& 반환, non-const 설정
{
	if(idx < 0 || idx >= arrlen)
	{
		cout << "Array index out of bound exception" << endl;
		exit(1);
	}
	
	return arr[idx];
}
```

이와 같이 **const 참조자로 참조하는 경우의 함수 호출**을 위해서 정의를 하고, 

**멤버 변수의 값을 변경할 수 있는 non-const 일반 함수를 정의함**으로써

보통 **두 가지가 동시에 정의되는 편**이다.

---

## 그 이외의 연산자 오버로딩 1.

### \[ new 연산자 오버로딩 ]

**new** 와 **delete** 도 **연산자이기 때문에 오버로딩이 가능**하다.
new 와 delete 를 C++ 문법을 구성하는 단순한 키워드 정도로만 인식하는 경우가 많은데, 이 둘은
분명한 연산자이다.
new 와 delete 의 오버로딩은 **앞서 보였던 연산자 오버로딩과는 많이 다르며**, 
둘을 이용해서 **클래스 별로 독특한 객체의 생성과정을 정의**할 수도 있다.

우선 **new 연산자** 에 대해 잠깐 살펴보자

**new 연산자가 하는 일**
- 메모리 공간의 할당
- 생성자 호출
- 할당하고자 하는 자료형에 맞게 반환된 주소 값의 형변환

이 세가지 역할 중 **메모리 공간 할당 작업**만 오버로딩이 가능하며, 나머지 **두 작업은 고정 역할**로 **오버로딩이 불가능**하다.

**객체의 생성과정은 다소 복잡**하니, **생성자의 호출과 반환된 주소 값의 형변환은 컴파일러가 담당**을 하고, **메모리 공간의 할당 부분**에 대해서만 **조율이 가능**하게 허락된 셈이다.

new 연산자의 오버로딩은 다음과 같이 오버로딩하도록 이미 약속되어 있다.
```c++
void* operator new(size_t size) { ... }
```
**반환형은 반드시 void 포인터 형**이어야 하고, **매게 변수형은 size_t** 이어야 한다.

그리고 이렇게 오버로딩된 함수는 **컴팡일러에 의해서 호출**이 이뤄진다.

예시 코드와 함께 설명을 이어가자면
```c++
class Point
{
private:
	int xpos, ypos;
public:
	Point(int x=0, int y=0) : xpos(x), ypos{y}
	{}
	friend ostream& operator<<(ostream& os, const Point& pos);
};

ostream& operator<<(ostream& os, const Point& pos)
{
	os << '[' << pos.xpos << ", " << pos.ypos << ']' << endl;
	
	return os;
}
```
Point  클래스 대상으로 new 연산자가 오버로딩된 상태에서 

```c++
Point* ptr = new Point(3, 4);
```
이와 같은 문장을 만나면, 컴파일러는 먼저 메모리 공간을 계산한다.
그리고 그 **크기가 계산되면 operator new() 함수를 호출하면서 계산된 크키의 값을 인자로 전달한다.**

여기서 중요한 것은 **크기 정보가 바이트 단위로 계산되어 전달**한다는 것이다.

따라서 우리는 대략 다음의 형태로 operator new() 함수를 정의해야 한다.
```c++
void* operator new(size_t size)
{
	voide *adr = new char[size];
	
	return adr;
}
```
컴파일러에 의해서, 필요한 **메모리 공간의 크기가 바이트 단위로 계산**되어 인자로 전달되니
**크기가 1바이트인 char 단위로 메모리 공간을 할당**해서 반환하였다.

이 코드가 operator new() 함수의 전부가 아니며, **operator new() 함수가 반드시 해야 할 일인 
'메모리 공간의 할당'을 각인** 시키기 위한 간단한 예제 코드이며, **우린 이 이상의 일을 하도록 operator() 함수를 정의**하고자 한다.

