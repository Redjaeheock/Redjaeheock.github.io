## 연산자 오버로딩 1

- C++ 에서는 함수뿐만 아니라 연산자도 오버로딩이 가능
- 연산자 오버로딩은 C++을 이해하는데 매우 중요한 요소
- 연산자 오버로딩시 함수 오버로딩과 마찬가지로 기존의 기능에서 추가적인 기능을 부여할 수 있음

C++ 에선 기존의 + 연산도 operator+ 라는 함수를 호출하는 걸로 인신
(기존 C 는 단순 연산)

### < operator+ 라는 이름의 함수 >
#### \[ operator+ 함수의 두가지 성격 ]

- 멤버 함수
	- 예시) **P1 + P2**
		- == P1 operator+ P2
		- 이 때, operator+ 는 **좌측 항을 대상으로 호출**하기로 약속됨
		- **== P1.operator+(p2)** 인 셈이다.
		- ![[Pasted image 20250909115918.png]]
- 전역 함수
	- 예시) **P1 + P2**
		- == **operator+(p1, p2)**
		- ![[Pasted image 20250909120135.png]]
- 전역 함수로 작성하던, 멤버 함수로 작성하던 **컴파일러는 알아서 구분 가능**

#### \[ 연산자 오버로딩 멤버 함수 버전 예시 코드]
``` c++
#include <iostream>
using namespace std;

class Point
{
private:
	int xpos, ypos;
public:
	Point(int x=0, int y=0) : xpos(x), ypos(y)
	{}
	void ShowPosition() const
	{
		cout << '[' << xpos << ", " << ypos << ']' << endl;
	}
	Point operator+(const Point &ref)    // operator+ 라는 함수이름
	{
		Point pos(xpos+ref.xpos, ypos+ref.ypos);
	}
};
```

``` c++
int main(void)
{
	Point pos1(3, 4);
	Point pos2(10, 20);
	//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	Point pos3 = pos1.operator+(pos2);
	//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	
	pos1.ShowPosition();
	pos2.ShowPosition();
	pos3.ShowPosition();
	
	// OR
	//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	Point pos4 = pos1 + pos2;
	//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	pos4.ShowPosition();
	
	return 0;
}
```
`실행 결과`
`[3, 4]`
`[10, 20]`
`[13, 24]`
`[13, 24]`

#### \[ 연산자 오버로딩 전역 함수 버전 예시 코드]
``` c++
#include <iostream>
using namespace std;

class Point
{
private:
	int xpos, ypos;
public:
	Point(int x=0, int y=0) : xpos(x), ypos(y)
	{}
	void ShowPosition() const
	{
		cout << '[' << xpos << ", " << ypos << ']' << endl;
	}
	friend Point operator+(const Point &pos1, const Point &pos2);
};

Point operator+(const Point &pos1, const Point &pos2)    // operator+ 라는 함수이름
{
	Point pos(pos1.xpos + pos2.xpos, pos1.ypos + pos2.ypos);
	
	return pos;
}
```

``` c++
int main(void)
{
	Point pos1(3, 4);
	Point pos2(10, 20);
	//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	Point pos3 = pos1 + pos2;
	//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	
	pos1.ShowPosition();
	pos2.ShowPosition();
	pos3.ShowPosition();
	
	return 0;
}
```
`실행 결과`
`[3, 4]`
`[10, 20]`
`[13, 24]`

---
## 오버로딩이 불가능한 연산자의 종류

##### 오버로딩을 허용하면 C++ 문법 체계에 반할 수 있는 부분들에 대해선 제약을 걸어둠

![[Pasted image 20250909122052.png]]

---
## 연산자 오버로딩시 주의사항

##### 본래의 의도를 벗어난 형태의 연산자 오버로딩을 지양
- 프로그램을 혼란스럽게 만들 수 있음
##### 연산자의 우선순위와 결합성을 바뀌지 않는다
- 따라서 이 둘을 고려해서 연산자를 오버로딩 하는 것을 지향
##### 매개 변수의 디폴트 값 설정이 불가능
- 매개 변수의 자료형에 따라서 호출되는 함수가 결정되므로 디폴트값 설정은 불가능
- 연산자는 이미 데이터 값이 존재할 경우에 적용되는 것으로 디폴트값 설정 자체가 무의미
##### 연산자의 순수 기능까지 뺏을 수는 없음
- 기본 자료형에 대한 연산은 오버로딩 불가능
- ``` c++
  int operator+(cont int num1, const int num2)
  {
	  return num1 * num2;
  }
  ```
- ↑정의 불가능한 형태

---
## 단항 연산자 오버로딩

#### \[ 증감연산자 ]

증감 연산자는 엄연히 **전치와 후치는 서로 다른 연산**으로 구분
- var++
- ++var
##### 멤버 함수로 사용시
- 전치 연산의 경우
	- operator++() - **인자없음**
- 후치 연산의 경우
	- operator++(int) - **int 를 명시**
##### 전역 함수로 사용시
- 전치 연산의 경우
	- operator++(var) - **필요인자 하나**
- 후치 연산의 경우
	- operatro++(var, int) - **필요인자 하나, int 추가 명시**

![[Pasted image 20250909140430.png]]
![[Pasted image 20250909140620.png]]
![[Pasted image 20250909140549.png]]
##### ※ 여기서 int 는 실제 값이 아닌 전치와 후치의 구분을 위한 단순 명시이다

---
## 증가, 감소 연산자 오버로딩

```c++
#include <ioctream>
using namespace std;

class Point
{
private:
	int xpos, ypos;
public:
	Point(int x=0, int y=0) : xpos(x), ypos(y)
	{}
	void ShowPosition() const
	{
		cout << '[' << xpos << ", " << ypos << ']' << endl;
	}
	Point& operator++()
	{
		xpos += 1;
		ypos += 1;
		
		return *this;
	}
	friend Point& operator--(Point &ref);
};
```

```c++
Point& operator--(Point &ref)
{
	ref.xpos -= 1;
	ref.ypos -= 1;
	
	return ref;
}
```

```c++
int main(void)
{
	Point pos(1, 2);
	++pos;
	pos.ShowPosition();
	
	--pos;
	pos.ShowPosition();
	
	++(++pos);
	pos.ShowPosition();
	
	--(--pos);
	pos.ShowPosition();
	
	return 0;
}
```
`실행결과`
`[2, 3]`
`[1, 2]`
`[3, 4]`
`[1, 2]`

![[Pasted image 20250909152527.png]]

---
## 전위 증가와 후위 증가의 구분

```c++
#include <iostream>
using namespace std;

class Point
{
private:
	int xpos, ypos;
public:
	Point(int x=0, int y=0) : xpos(x), ypos(y)
	{}
	void ShowPosition() const
	{
		cout << '[' << xpos << ", " << ypos << ']' << endl;
	}
	Point& operator++()
	{
		xpos += 1;
		ypos += 1;
		
		return *this;
	}
	const Point operator++(int)
	{
		const Point retobj(xpos, ypos);
		// == const Point retobj(*this);
		
		xpos += 1;
		ypos += 1;
		
		return retobj;
	}
	friend Point& operator--(Point& ref);
	friend const Point operator--(Point& ref, int);
};
```

```c++
Point& operator--(Point &ref) // 전위감소
{
	ref.xpos -= 1;
	ref.ypos -= 1;
	
	return ref;
}
```

```c++
const Point operator--(Point& ref, int) //후위 감소
{
	const Point retobj(ref); //const 객체
	
	ref.xpos -= 1;
	ref.ypos -= 1;
	
	return retobj;
}
```

```c++
int main(void)
{
	Point pos(3, 5);
	Point copy;
	
	copy = pos--;
	copy.ShowPosition();
	pos.ShowPosition();
	
	copy = pos++;
	copy.ShowPosition()
	pos.ShowPosition();
	
	++(++pos);
	pos.ShowPosition();

	return 0;
}
```
`실행결과`
`[3, 5]`
`[2, 4]`
`[2, 4]`
`[3, 5]`

![[Pasted image 20250909143419.png]]
![[Pasted image 20250909143430.png]]

이미지의 두 부분에서 **내부에서 객체 복사**를 통해 

후위연산의 **연산행 이후의 적용 메커니즘 구현**을 도와준다

```c
int a = 1;
printf("%d", a++);
printf("%d", a);
```
`실행결과`
`1`
`2`

---
## 반환형에서의 const 선언과 const 객체

#### \[ const ]

- const 객체는 멤버 변수의 변경이 불가능한 객체
- const 객체는 const 참조자로만 참조가 가능
- const 객체를 대상으로는 const 함수만 호출 가능

후위 증가 연산자 오버로딩에 **const 반환형을 사용한 궁극적인 이유**
- ##### 후위 증가 연산의 연속 사용을 방지하기 위해서

```c++
int a = 1;

++(++a);  // 유효
(a++)++;  // 무효
```
C++ 자체에선 **후위 증가 연산의 연속 사용을 허용 하지 않기 때문**에, 
룰을 적용하기 위한 장치로 const 객체를 사용

전위 증가 연산의 경우 C 에선 연산이 적용이 안되지만
C++ 에선 객체의 참조 반환을 통해 이뤄지기 때문에 사용 가능함

**lvalue**: `“주소가 있는(메모리 위치를 가리키는) 표현식”.
- `보통 름이 있는 것**들: 변수 `a`, 참조로 받은 매개변수, `\*p`, `arr\[i]`, `s.x` 등.
- `대개 `&`(address-of) 를 붙일 수 있고, 수정 가능한 lvalue이면 왼쪽값으로 대입 가능.`

즉, 전치 $\cdot$ 후치 연산 과정에서 피연산자가 lvalue 이냐 아니냐에 따라 적용 여부가 달라진다

![[Pasted image 20250910194002.png]]

