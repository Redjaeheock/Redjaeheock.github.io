## 연산자 오버로딩2

### < 반드시 해야 하는 대입 연산자의 오버로딩 >

연산자 오버로딩의 필요성에 대한 내용을 다루기 전에 __복사 생성자__ 에 대해서 다시 다뤄본다면

__복사 생성자__
- 정의하지 않으면 __Default 복사 생성자__ 가 삽입됨(컴파일러)
- Default 복사 생성자는 멤버 대 멤버의 복사(얕은복사)를 진행함
- 생성자 내에서 동적 할당하거나 깊은 복사가 필요할 경우 작성자가 직접 정의 해야함


다음은 __대입 연산자__ 의 대표적인 특성이다


__대입 연산자__
- 정의하지 않으면 __Default 대입 연산자__ 가 삽입됨(컴파일러)
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

__pos2 를  '초기화 하는 과정' 에서 이미 생성된 객체인 pos1__ 을 사용한 것이다.


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

__이미 생성된 객체 pos2 를 이미 생성된 객체 pos1 로 '대입'__ 을 하는 것이다.

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

출력이 하나가 됐든 두 개가 되었든 __객체의 소멸 과정__ 에서 문제가 발생했단 것을 짐작 할 수 있다.

__operator+()__ 를 정의하지 않았기 때문에, __Default 대입 연산자__ 가 삽입되면서
__Person 생성자__ 내부에서 name 에 __동적할당__ 된 메모리가 __얕은복사__ 가 된채로 man1 과 man2 가
같은 주소를 공유하고 잇는 상태인데

여기서 __man1이 먼저 소멸__ 되면서 __name 의 메모리가 해제__ 되면서, 

이후 __같은 주소를 소유하던 man2__  의 소멸자 입장에선 해제할 메모리가 없어 
__Double free__ 가 유도되는 상황이 되버린다

이는 다른 챕터에서 복사 생성자의 __얕은 복사에서의 문제점__ 과 동일한 내용임을 알 수 있다.

이를 방지하기 위해서 __깊은 복사가 필요__ 하고 그로 인해
__반드시 대입 연산자의 오버로딩__ 의 정의가 필요한 것이다.

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

__Second 객체의 대입 연산자를 주석__ 처리한 후 객체끼리의 대입을 실행했을 때,

Second 클래스에선 대입 연산자가 없으니 __컴파일러가 삽입한 Default 대입 연산자__ 가 실행이 되고, 
이 Default 대입 연산자가 __First 클래스의 대입 연산자__ 까지 호출한다는 것을 유추할 수 있는데,

__First::operator=() 의 출력문__ 이 출력되어 있고, 
__scpy.ShowData()__ 의 결과로 __scpy 의 상위 영역의 멤버 값 (0, 0) 이 (111, 222) 로 출력__ 된 부분이 유추를 가능케 하는 대목이다


하지만 우린 __대입 연산자를 직접 정의하는 것을 지향__ 하기 때문에 __scpy 의 주석을 해제__ 하고 실행한다면 다음과 같은 결과를 얻을 수 있다.

```c++
---------------------------------------------------------------------------------
```
```
Second& operator=()
0, 0
333, 444
```
직접 정의한 코드에선 __First 클래스 영역 멤버 변수__ 에 대한 내용을 다루지 않았기 때문에,
자연스럽게 __Second 클래스 영역 멤버 변수만 대입__ 처리가 되어 있는 것을 볼 수 있다.

우린 그럼 c++ 의 자연스러운 동작을 유지하기 위해 __First 클래스 영역 멤버 변수의 대입__ 을 유도하는 것 또한 책임을 지어야 한다.

그렇다고 __First 클래스의 멤버를 직접 접근해서 대입__ 하는 것은 c++ 의 원칙에 맞지 않으니, 
아래의 코드 처럼 __First 클래스의 operator=() 를 호출__ 하는 것이 바람직 한 것으로 사료된다.

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


여기서 중요한 점은 __멤버의 동적 할당이나 특별히 적접 정의할 내용__ 이 없다면, 
__상속과 피상속 클래스에서 대입 연산자의 직접 정의 없이 컴파일러의 Default 대입 연산자__ 에게 맡기는 게
작성자의 실수보다 조금 더 안전한 동작을 기대할 수 있다는 것이다.

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

결과에서 알 수 있듯이 __이니셜라이저__ 를 통한 값은 __객체 생성과 동시에__ 한 번에 정리되는 한 편,
__대입연산__ 을 통한 값은 __객체 생성 후 대입 연산자를 별도 호출__ 함으로써 두 번의 동작이 이뤄지는 것을 볼 수 있다.

이처럼 객체의 초기와 과정에선 가능한 __이니셜라이저를 상요할 것을 지향__ 하는 바이다.


