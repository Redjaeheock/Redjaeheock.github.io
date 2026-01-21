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