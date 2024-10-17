---
layout: single
title:  "Google_drive_to_web"
---

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Make MyShell Project"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## bash manual traslation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Bash Features\n",
    "\n",
    "This text is a brief description of the features that are present in the Bash shell (version 5.2, 19 September 2022). The Bash home page is http://www.gnu.org/software/bash/.\n",
    "\n",
    "This is Edition 5.2, last updated 19 September 2022, of The GNU Bash Reference Manual, for Bash, Version 5.2.\n",
    "\n",
    "Bash contains features that appear in other popular shells, and some features that only appear in Bash. Some of the shells that Bash has borrowed concepts from are the Bourne Shell (sh), the Korn Shell (ksh), and the C-shell (csh and its successor, tcsh). The following menu breaks the features up into categories, noting which features were inspired by other shells and which are specific to Bash.\n",
    "\n",
    "This manual is meant as a brief introduction to features found in Bash. The Bash manual page should be used as the definitive reference on shell behavior."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- 이것은 bash 버전 5.2. 에 대한 bash 참고 메뉴얼의 2022년 9월 19 일 마지막 업데이트된 편집본이다.\n",
    "\n",
    "- bash 는 다른 shell 들(sh, ksh, sch, tcsh)이 나타내는 특징을 가지고 있고, bash 만의 특징도 가지고 있다.\n",
    "- 다음 메뉴는 특징들을 카테고리별로 나눈다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "present - 선물하다/제공하다/발표하다/현재 (라틴어 'praesentare'에서 유래되었으며, 'prae-' 라는 '앞에' 와 'esse' '존재하다' 에서 파생되었으며 '어떤 것을 앞에 존재하게 하다; 라는 의미로 '앞에 놓다' 라는 뜻으로 사용되었다. 이후 무언가를 '앞에 놓고 보여주거나 제공하는 행위'로 의미가 확장되었으며 무언가를 '제시하다' / '제공하다' 라는 의미로 발전하게 되었다. 또한 '존재하다' 라는 것이 '현재 존재하는 것' 으로 '지금 이 순간/현재' 라는 뜻으로 같이 사용하게 되었다.)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 1.2 What is a shell?\n",
    "Bash is the shell, or command language interpreter, for the GNU operating system. The name is an acronym for the ‘Bourne-Again SHell’, a pun on Stephen Bourne, the author of the direct ancestor of the current Unix shell sh, which appeared in the Seventh Edition Bell Labs Research version of Unix.\n",
    "\n",
    "Bash is largely compatible with sh and incorporates useful features from the Korn shell ksh and the C shell csh. It is intended to be a conformant implementation of the IEEE POSIX Shell and Tools portion of the IEEE POSIX specification (IEEE Standard 1003.1). It offers functional improvements over sh for both interactive and programming use.\n",
    "\n",
    "While the GNU operating system provides other shells, including a version of csh, Bash is the default shell. Like other GNU software, Bash is quite portable. It currently runs on nearly every version of Unix and a few other operating systems - independently-supported ports exist for MS-DOS, OS/2, and Windows platforms."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " Bash는 GNU 운영 체제의 쉘이자 명령어 언어 해석기입니다. 이름은 ‘Bourne-Again SHell’의 약자로, 현재의 Unix 쉘(sh)의 직접적인 전신을 작성한 Stephen Bourne의 이름을 딴 말장난입니다. 이 쉘은 Unix의 Bell Labs Research 7판 버전에 처음 등장했습니다.\n",
    "\n",
    "Bash는 sh과 대체로 호환되며, Korn 쉘(ksh)과 C 쉘(csh)에서 유용한 기능들을 통합했습니다. Bash는 IEEE POSIX 사양(IEEE 표준 1003.1)의 POSIX 쉘 및 도구 부분을 준수하는 구현을 목표로 합니다. 이는 sh보다 상호작용 및 프로그래밍 사용 모두에 있어 기능적인 개선을 제공합니다.\n",
    "\n",
    "GNU 운영 체제는 csh 버전을 포함하여 다른 쉘도 제공하지만, 기본 쉘은 Bash입니다. 다른 GNU 소프트웨어들처럼 Bash는 매우 이식성이 뛰어납니다. 현재 거의 모든 Unix 버전과 몇 가지 다른 운영 체제에서 실행되며, MS-DOS, OS/2, Windows 플랫폼에 대한 독립적인 지원 포트도 존재합니다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 1.1 What is Bash?\n",
    "At its base, a shell is simply a macro processor that executes commands. The term macro processor means functionality where text and symbols are expanded to create larger expressions.\n",
    "\n",
    "A Unix shell is both a command interpreter and a programming language. As a command interpreter, the shell provides the user interface to the rich set of GNU utilities. The programming language features allow these utilities to be combined. Files containing commands can be created, and become commands themselves. These new commands have the same status as system commands in directories such as /bin, allowing users or groups to establish custom environments to automate their common tasks.\n",
    "\n",
    "Shells may be used interactively or non-interactively. In interactive mode, they accept input typed from the keyboard. When executing non-interactively, shells execute commands read from a file.\n",
    "\n",
    "A shell allows execution of GNU commands, both synchronously and asynchronously. The shell waits for synchronous commands to complete before accepting more input; asynchronous commands continue to execute in parallel with the shell while it reads and executes additional commands. The redirection constructs permit fine-grained control of the input and output of those commands. Moreover, the shell allows control over the contents of commands’ environments.\n",
    "\n",
    "Shells also provide a small set of built-in commands (builtins) implementing functionality impossible or inconvenient to obtain via separate utilities. For example, cd, break, continue, and exec cannot be implemented outside of the shell because they directly manipulate the shell itself. The history, getopts, kill, or pwd builtins, among others, could be implemented in separate utilities, but they are more convenient to use as builtin commands. All of the shell builtins are described in subsequent sections.\n",
    "\n",
    "While executing commands is essential, most of the power (and complexity) of shells is due to their embedded programming languages. Like any high-level language, the shell provides variables, flow control constructs, quoting, and functions.\n",
    "\n",
    "Shells offer features geared specifically for interactive use rather than to augment the programming language. These interactive features include job control, command line editing, command history and aliases. Each of these features is described in this manual."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- 기본적으로 shell 은 단순히 명령어들을 실행하는 매크로 프로세서 이다. 매크로 프로세서 용어의 의미는 \n",
    "\n",
    "기본적으로 쉘은 단순히 명령을 실행하는 매크로 프로세서입니다. 매크로 프로세서라는 용어는 텍스트와 기호가 확장되어 더 큰 표현을 생성하는 기능을 의미합니다.\n",
    "\n",
    "유닉스 쉘은 명령어 해석기이자 프로그래밍 언어입니다. 명령어 해석기로서 쉘은 GNU 유틸리티의 풍부한 기능 세트에 대한 사용자 인터페이스를 제공합니다. 프로그래밍 언어 기능을 통해 이러한 유틸리티들을 결합할 수 있으며, 명령을 포함한 파일을 작성하여 자체적으로 명령이 될 수 있습니다. 이러한 새로운 명령들은 /bin 같은 디렉토리에 있는 시스템 명령들과 동일한 지위를 가지며, 사용자나 그룹이 반복적인 작업을 자동화하기 위해 맞춤형 환경을 구축할 수 있게 합니다.\n",
    "\n",
    "쉘은 대화형 모드 또는 비대화형 모드로 사용할 수 있습니다. 대화형 모드에서는 키보드로 입력된 명령을 받아들이며, 비대화형 모드에서는 파일에서 읽은 명령을 실행합니다.\n",
    "\n",
    "쉘은 GNU 명령을 동기적 및 비동기적으로 실행할 수 있습니다. 동기적 명령의 경우, 쉘은 명령이 완료될 때까지 기다린 후에야 추가 입력을 받으며, 비동기적 명령은 쉘이 추가 명령을 읽고 실행하는 동안 병렬로 계속 실행됩니다. 리다이렉션 구문을 통해 이러한 명령의 입력과 출력을 세밀하게 제어할 수 있습니다. 또한, 쉘은 명령의 환경 내용을 제어할 수 있게 합니다.\n",
    "\n",
    "쉘은 별도의 유틸리티로는 구현할 수 없거나 불편한 기능을 제공하는 소수의 내장 명령어(builtins)를 제공합니다. 예를 들어, cd, break, continue, exec는 쉘 자체를 직접 조작하기 때문에 쉘 외부에서는 구현할 수 없습니다. 반면, history, getopts, kill, pwd와 같은 내장 명령어들은 별도의 유틸리티로도 구현할 수 있지만, 내장 명령어로 사용하는 것이 더 편리합니다. 모든 쉘 내장 명령어들은 이후 섹션에서 설명됩니다.\n",
    "\n",
    "명령을 실행하는 것이 필수적이긴 하지만, 쉘의 대부분의 강력함(및 복잡성)은 내장된 프로그래밍 언어에 기인합니다. 고급 언어처럼 쉘은 변수, 흐름 제어 구문, 인용(quoting), 함수 등의 기능을 제공합니다.\n",
    "\n",
    "쉘은 프로그래밍 언어를 확장하기보다는 대화형 사용을 위해 특별히 설계된 기능을 제공합니다. 이러한 대화형 기능에는 작업 제어, 명령 줄 편집, 명령 기록, 별칭 등이 포함됩니다. 이들 각각의 기능은 이 매뉴얼에서 설명됩니다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "execute - 실행하다, 처형하다\n",
    "(라틴어 exsequi 에서 유래되었으며, '뒤따라가다' 또는 '이행하다' 라는 의미로 사용되었다가 중세 영어로 넘어오면서 주로 '법적 처벌을 이행하다(사형을 집행하다)'는 의미로 사용되었다. 이후 다시 '명령이나 계획을 이행하다' 라는 뜻으로 같이 사용되다 최종적으로 '실행하다' 라는 뜻과 같이 쓰이게 되었다. \n",
    "이행하다 -> carry out : 어떤 지시, 계쇡, 명령을 따르거나 수행하는 것을 의미하며, '결과보다는 행위 자체'에 더 초점을 맞춘다.\n",
    "실행하다 -> run/perform : 특정 작업이나 행동을 실제적으로 수행하는 것을 의미하며, '결과를 만드는 것'에 초점을 맞춘다. \n",
    "\n",
    "term - 용어, 기간\n",
    "(라틴어 terminus 에서 유래되었으며, '특정 개념이나 대상의 정의나 경계를 명확하게 설명하는 단어'를 가리키는데 사용되었으며, 이로 인해 어떤 것이 '끝나는 시점 또는 정해진 경계(범주)'를 가리키는 용도로도 사용되기도 했다. 후에 정의라는 면에서는 그대로 사용되었고, 범주로 사용하던 뜻은 확장되어 '기간'이란 뜻으로 사용되고 있다)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2 Definitions\n",
    "These definitions are used throughout the remainder of this manual.\n",
    "\n",
    "##### POSIX\n",
    "A family of open system standards based on Unix. Bash is primarily concerned with the Shell and Utilities portion of the POSIX 1003.1 standard.\n",
    "\n",
    "##### blank\n",
    "A space or tab character.\n",
    "\n",
    "##### builtin\n",
    "A command that is implemented internally by the shell itself, rather than by an executable program somewhere in the file system.\n",
    "\n",
    "##### control operator\n",
    "A token that performs a control function. It is a newline or one of the following: ‘||’, ‘&&’, ‘&’, ‘;’, ‘;;’, ‘;&’, ‘;;&’, ‘|’, ‘|&’, ‘(’, or ‘)’.\n",
    "\n",
    "##### exit status\n",
    "The value returned by a command to its caller. The value is restricted to eight bits, so the maximum value is 255.\n",
    "\n",
    "##### field\n",
    "A unit of text that is the result of one of the shell expansions. After expansion, when executing a command, the resulting fields are used as the command name and arguments.\n",
    "\n",
    "##### filename\n",
    "A string of characters used to identify a file.\n",
    "\n",
    "##### job\n",
    "A set of processes comprising a pipeline, and any processes descended from it, that are all in the same process group.\n",
    "\n",
    "##### job control\n",
    "A mechanism by which users can selectively stop (suspend) and restart (resume) execution of processes.\n",
    "\n",
    "##### metacharacter\n",
    "A character that, when unquoted, separates words. A metacharacter is a space, tab, newline, or one of the following characters: ‘|’, ‘&’, ‘;’, ‘(’, ‘)’, ‘<’, or ‘>’.\n",
    "\n",
    "##### name\n",
    "A word consisting solely of letters, numbers, and underscores, and beginning with a letter or underscore. Names are used as shell variable and function names. Also referred to as an identifier.\n",
    "\n",
    "##### operator\n",
    "A control operator or a redirection operator. See Redirections, for a list of redirection operators. Operators contain at least one unquoted metacharacter.\n",
    "\n",
    "##### process group\n",
    "A collection of related processes each having the same process group ID.\n",
    "\n",
    "##### process group ID\n",
    "A unique identifier that represents a process group during its lifetime.\n",
    "\n",
    "##### reserved word\n",
    "A word that has a special meaning to the shell. Most reserved words introduce shell flow control constructs, such as for and while.\n",
    "\n",
    "##### return status\n",
    "A synonym for exit status.\n",
    "\n",
    "##### signal\n",
    "A mechanism by which a process may be notified by the kernel of an event occurring in the system.\n",
    "\n",
    "##### special builtin\n",
    "A shell builtin command that has been classified as special by the POSIX standard.\n",
    "\n",
    "##### token\n",
    "A sequence of characters considered a single unit by the shell. It is either a word or an operator.\n",
    "\n",
    "##### word\n",
    "A sequence of characters treated as a unit by the shell. Words may not include unquoted metacharacters."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2 정의\n",
    "이 정의들은 이 매뉴얼의 나머지 부분에서 사용됩니다.\n",
    "\n",
    "##### POSIX\n",
    "유닉스를 기반으로 한 오픈 시스템 표준의 집합입니다. Bash는 주로 POSIX 1003.1 표준의 쉘 및 유틸리티 부분과 관련이 있습니다.\n",
    "\n",
    "##### blank\n",
    "공백 또는 탭 문자입니다.\n",
    "\n",
    "##### builtin\n",
    "쉘 자체에서 내부적으로 구현된 명령어로, 파일 시스템 어딘가의 실행 파일로 구현된 것이 아닙니다.\n",
    "\n",
    "##### control operator\n",
    "제어 기능을 수행하는 토큰입니다. 이는 줄 바꿈 문자이거나 다음 중 하나입니다: ‘||’, ‘&&’, ‘&’, ‘;’, ‘;;’, ‘;&’, ‘;;&’, ‘|’, ‘|&’, ‘(’, 또는 ‘)’입니다.\n",
    "\n",
    "##### exit status\n",
    "명령이 호출자에게 반환하는 값입니다. 이 값은 8비트로 제한되므로 최대 값은 255입니다.\n",
    "\n",
    "##### field\n",
    "쉘 확장의 결과로 생성된 텍스트의 단위입니다. 확장 후 명령을 실행할 때, 생성된 필드는 명령 이름과 인수로 사용됩니다.\"\n",
    "\n",
    "##### filename\n",
    "파일을 식별하는 데 사용되는 문자열입니다.\n",
    "\n",
    "##### job\n",
    "파이프라인을 구성하는 일련의 프로세스와 그로부터 파생된 모든 프로세스들이 동일한 프로세스 그룹에 속하는 집합입니다.\n",
    "\n",
    "##### job control\n",
    "사용자가 프로세스의 실행을 선택적으로 중지(일시 정지)하거나 다시 시작(재개)할 수 있는 메커니즘입니다.\n",
    "\n",
    "##### metacharacter\n",
    "인용되지 않았을 때 단어를 구분하는 문자입니다. 메타문자는 공백, 탭, 줄 바꿈 또는 다음의 문자 중 하나입니다: ‘|’, ‘&’, ‘;’, ‘(’, ‘)’, ‘<’, 또는 ‘>’.\n",
    "\n",
    "##### name\n",
    "문자와 숫자, 밑줄로만 구성된 단어이며, 문자 또는 밑줄로 시작합니다. 이름은 쉘 변수와 함수 이름으로 사용됩니다. 식별자(identifier)라고도 합니다.\n",
    "\n",
    "##### operator\n",
    "제어 연산자 또는 리다이렉션 연산자를 의미합니다. 리다이렉션 연산자의 목록은 리다이렉션 섹션을 참조하십시오. 연산자는 최소 하나 이상의 인용되지 않은 메타문자를 포함합니다.\n",
    "\n",
    "##### process group\n",
    "같은 프로세스 그룹 ID를 가진 관련된 프로세스들의 집합입니다.\n",
    "\n",
    "##### process group ID\n",
    "프로세스 그룹의 생애 동안 이를 나타내는 고유한 식별자입니다.\n",
    "\n",
    "##### reserved word\n",
    "쉘에서 특별한 의미를 가지는 단어입니다. 대부분의 예약어는 for와 while과 같은 쉘의 흐름 제어 구문을 도입합니다.\n",
    "\n",
    "##### return status\n",
    "exit status의 동의어입니다.\n",
    "\n",
    "##### signal\n",
    "프로세스가 커널에 의해 시스템에서 발생하는 이벤트를 통지받을 수 있는 메커니즘입니다.\n",
    "\n",
    "##### special builtin\n",
    "POSIX 표준에 의해 특별히 분류된 쉘의 내장 명령어입니다.\n",
    "\n",
    "##### token\n",
    "쉘에 의해 하나의 단위로 간주되는 문자들의 연속입니다. 이는 단어 또는 연산자일 수 있습니다.\n",
    "\n",
    "##### word\n",
    "쉘에 의해 하나의 단위로 취급되는 문자들의 연속입니다. 단어에는 인용되지 않은 메타문자가 포함될 수 없습니다.\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Bash is an acronym for ‘Bourne-Again SHell’. The Bourne shell is the traditional Unix shell originally written by Stephen Bourne. All of the Bourne shell builtin commands are available in Bash, The rules for evaluation and quoting are taken from the POSIX specification for the ‘standard’ Unix shell.\n",
    "\n",
    "This chapter briefly summarizes the shell’s ‘building blocks’: commands, control structures, shell functions, shell parameters, shell expansions, redirections, which are a way to direct input and output from and to named files, and how the shell executes commands."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Bash는 ‘Bourne-Again SHell’의 약자입니다. Bourne 쉘은 Stephen Bourne이 작성한 전통적인 유닉스 쉘입니다. 모든 Bourne 쉘의 내장 명령어들이 Bash에서도 사용 가능하며, 평가와 인용에 관한 규칙은 ‘표준’ 유닉스 쉘에 대한 POSIX 사양을 따릅니다.\n",
    "\n",
    "이 장에서는 쉘의 ‘구성 요소’들을 간단히 요약합니다: 명령어, 제어 구조, 쉘 함수, 쉘 매개변수, 쉘 확장, 리다이렉션(지정된 파일에서 입력을 받고 출력하는 방법), 그리고 쉘이 명령을 실행하는 방식을 설명합니다.\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "When the shell reads input, it proceeds through a sequence of operations. If the input indicates the beginning of a comment, the shell ignores the comment symbol (‘#’), and the rest of that line.\n",
    "\n",
    "Otherwise, roughly speaking, the shell reads its input and divides the input into words and operators, employing the quoting rules to select which meanings to assign various words and characters.\n",
    "\n",
    "The shell then parses these tokens into commands and other constructs, removes the special meaning of certain words or characters, expands others, redirects input and output as needed, executes the specified command, waits for the command’s exit status, and makes that exit status available for further inspection or processing."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "\"쉘이 입력을 읽을 때, 일련의 작업을 통해 처리합니다. 입력이 주석의 시작을 나타내면, 쉘은 주석 기호(‘#’)와 해당 줄의 나머지 부분을 무시합니다.\n",
    "\n",
    "그 외의 경우, 대략적으로 말해, 쉘은 입력을 읽고 입력을 단어와 연산자로 나누며, 다양한 단어와 문자에 의미를 부여하기 위해 인용 규칙을 사용합니다.\n",
    "\n",
    "그 후, 쉘은 이러한 토큰을 명령어와 다른 구조로 파싱하고, 특정 단어나 문자의 특별한 의미를 제거하며, 다른 것들을 확장하고, 필요에 따라 입력과 출력을 리다이렉션합니다. 그런 다음 지정된 명령을 실행하고, 명령의 종료 상태를 기다리며, 그 종료 상태를 추가적인 검사나 처리에 사용할 수 있도록 제공합니다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The following is a brief description of the shell’s operation when it reads and executes a command. Basically, the shell does the following:\n",
    "\n",
    "1. Reads its input from a file (see Shell Scripts), from a string supplied as an argument to the -c invocation option (see Invoking Bash), or from the user’s terminal.\n",
    "2. Breaks the input into words and operators, obeying the quoting rules described in Quoting. These tokens are separated by metacharacters. Alias expansion is performed by this step (see Aliases).\n",
    "3. Parses the tokens into simple and compound commands (see Shell Commands).\n",
    "4. Performs the various shell expansions (see Shell Expansions), breaking the expanded tokens into lists of filenames (see Filename Expansion) and commands and arguments.\n",
    "5. Performs any necessary redirections (see Redirections) and removes the redirection operators and their operands from the argument list.\n",
    "6. Executes the command (see Executing Commands).\n",
    "7. Optionally waits for the command to complete and collects its exit status (see Exit Status)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "다음은 쉘이 명령을 읽고 실행할 때의 작동 방식을 간략하게 설명한 것입니다. 기본적으로, 쉘은 다음과 같은 작업을 수행합니다:\n",
    "\n",
    "파일(참고: 쉘 스크립트), -c 호출 옵션의 인수로 제공된 문자열(참고: Bash 호출), 또는 사용자의 터미널에서 입력을 읽습니다.\n",
    "입력을 인용 규칙에 따라 단어와 연산자로 나누어 토큰화합니다(참고: 인용). 이러한 토큰은 메타문자에 의해 분리됩니다. 이 단계에서 별칭 확장이 수행됩니다(참고: 별칭).\n",
    "토큰을 단순 및 복합 명령으로 파싱합니다(참고: 쉘 명령).\n",
    "다양한 쉘 확장을 수행하고(참고: 쉘 확장), 확장된 토큰을 파일 이름 목록(참고: 파일 이름 확장), 명령어 및 인수로 나눕니다.\n",
    "필요한 리다이렉션을 수행하고(참고: 리다이렉션), 리다이렉션 연산자와 해당 피연산자를 인수 목록에서 제거합니다.\n",
    "명령을 실행합니다(참고: 명령 실행).\n",
    "선택적으로 명령이 완료되기를 기다리고, 그 종료 상태를 수집합니다(참고: 종료 상태).\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Quoting is used to remove the special meaning of certain characters or words to the shell. Quoting can be used to disable special treatment for special characters, to prevent reserved words from being recognized as such, and to prevent parameter expansion.\n",
    "\n",
    "Each of the shell metacharacters (see Definitions) has special meaning to the shell and must be quoted if it is to represent itself. When the command history expansion facilities are being used (see History Expansion), the history expansion character, usually ‘!’, must be quoted to prevent history expansion. See Bash History Facilities, for more details concerning history expansion.\n",
    "\n",
    "There are three quoting mechanisms: the escape character, single quotes, and double quotes."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "인용(Quoting)은 쉘에서 특정 문자나 단어의 특별한 의미를 제거하기 위해 사용됩니다. 인용은 특수 문자의 특별한 처리를 비활성화하거나, 예약어가 그러한 의미로 인식되는 것을 방지하며, 매개변수 확장을 방지하기 위해 사용할 수 있습니다.\n",
    "\n",
    "쉘 메타문자(참고: 정의)는 쉘에 특별한 의미를 가지며, 그 자체를 표현하려면 인용해야 합니다. 명령 기록 확장 기능이 사용될 때(참고: 기록 확장), 기록 확장 문자(보통 ‘!’)는 기록 확장을 방지하기 위해 인용해야 합니다. 기록 확장에 관한 자세한 내용은 Bash 기록 기능을 참조하세요.\n",
    "\n",
    "세 가지 인용 메커니즘이 있습니다: 이스케이프 문자, 작은따옴표, 큰따옴표."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "3.1.2.1 Escape Character\n",
    "A non-quoted backslash ‘\\’ is the Bash escape character. It preserves the literal value of the next character that follows, with the exception of newline. If a \\newline pair appears, and the backslash itself is not quoted, the \\newline is treated as a line continuation (that is, it is removed from the input stream and effectively ignored)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\"3.1.2.1 이스케이프 문자\n",
    "따옴표로 감싸지지 않은 백슬래시 ‘\\’는 Bash의 이스케이프 문자입니다. 이는 다음에 오는 문자의 리터럴 값을 유지합니다. 단, 줄 바꿈(newline)은 예외입니다. 만약 \\newline 쌍이 나타나고 백슬래시 자체가 인용되지 않은 경우, \\newline은 줄 계속(line continuation)으로 처리됩니다(즉, 입력 스트림에서 제거되어 효과적으로 무시됩니다).\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "3.1.2.2 Single Quotes\n",
    "Enclosing characters in single quotes (‘'’) preserves the literal value of each character within the quotes. A single quote may not occur between single quotes, even when preceded by a backslash."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\"3.1.2.2 작은따옴표\n",
    "문자를 작은따옴표(‘'’)로 감싸면 따옴표 안의 각 문자의 리터럴 값이 그대로 유지됩니다. 작은따옴표는 작은따옴표 사이에 올 수 없으며, 백슬래시가 앞에 있어도 예외가 아닙니다.\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "3.1.2.3 Double Quotes\n",
    "Enclosing characters in double quotes (‘\"’) preserves the literal value of all characters within the quotes, with the exception of ‘$’, ‘’, ‘\\’, and, when history expansion is enabled, ‘!’. When the shell is in POSIX mode (see Bash POSIX Mode), the ‘!’ has no special meaning within double quotes, even when history expansion is enabled. The characters ‘$’ and ‘’ retain their special meaning within double quotes (see Shell Expansions). The backslash retains its special meaning only when followed by one of the following characters: ‘$’, ‘’, ‘\"’, ‘\\’, or newline. Within double quotes, backslashes that are followed by one of these characters are removed. Backslashes preceding characters without a special meaning are left unmodified. A double quote may be quoted within double quotes by preceding it with a backslash. If enabled, history expansion will be performed unless an ‘!’ appearing in double quotes is escaped using a backslash. The backslash preceding the ‘!’ is not removed.\n",
    "\n",
    "The special parameters ‘*’ and ‘@’ have special meaning when in double quotes (see Shell Parameter Expansion)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "3.1.2.3 큰따옴표\n",
    "문자를 큰따옴표(‘\"’)로 감싸면 따옴표 안의 모든 문자들의 리터럴 값이 유지되지만, ‘$’, ‘’, ‘\\’, 그리고 기록 확장이 활성화된 경우 ‘!’는 예외입니다. 쉘이 POSIX 모드일 때(참고: Bash POSIX 모드), 기록 확장이 활성화되어 있어도 큰따옴표 안에서는 ‘!’가 특별한 의미를 가지지 않습니다. 큰따옴표 안에서는 ‘$’와 ‘’는 여전히 특별한 의미를 가집니다(참고: 쉘 확장). 백슬래시는 다음 문자들 중 하나가 뒤따를 때만 특별한 의미를 유지합니다: ‘$’, ‘`’, ‘\"’, ‘\\’, 또는 줄 바꿈. 큰따옴표 안에서 이 문자들 중 하나가 백슬래시 뒤에 오면, 백슬래시는 제거됩니다. 특별한 의미가 없는 문자 앞에 오는 백슬래시는 수정되지 않고 그대로 남습니다. 큰따옴표 안에 있는 큰따옴표는 앞에 백슬래시를 붙여 인용할 수 있습니다. 기록 확장이 활성화된 경우, 큰따옴표 안에 있는 ‘!’는 백슬래시로 이스케이프되지 않으면 기록 확장이 수행됩니다. ‘!’ 앞에 있는 백슬래시는 제거되지 않습니다.\n",
    "\n",
    "특별 매개변수 ‘*’와 ‘@’는 큰따옴표 안에서 특별한 의미를 가집니다(참고: 쉘 매개변수 확장)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "3.1.2.4 ANSI-C Quoting\n",
    "Character sequences of the form $’string’ are treated as a special kind of single quotes. The sequence expands to string, with backslash-escaped characters in string replaced as specified by the ANSI C standard. Backslash escape sequences, if present, are decoded as follows:\n",
    "\n",
    "\\a\n",
    "alert (bell)\n",
    "\n",
    "\\b\n",
    "backspace\n",
    "\n",
    "\\e\n",
    "\\E\n",
    "an escape character (not ANSI C)\n",
    "\n",
    "\\f\n",
    "form feed\n",
    "\n",
    "\\n\n",
    "newline\n",
    "\n",
    "\\r\n",
    "carriage return\n",
    "\n",
    "\\t\n",
    "horizontal tab\n",
    "\n",
    "\\v\n",
    "vertical tab\n",
    "\n",
    "\\\\\n",
    "backslash\n",
    "\n",
    "\\'\n",
    "single quote\n",
    "\n",
    "\\\"\n",
    "double quote\n",
    "\n",
    "\\?\n",
    "question mark\n",
    "\n",
    "\\nnn\n",
    "the eight-bit character whose value is the octal value nnn (one to three octal digits)\n",
    "\n",
    "\\xHH\n",
    "the eight-bit character whose value is the hexadecimal value HH (one or two hex digits)\n",
    "\n",
    "\\uHHHH\n",
    "the Unicode (ISO/IEC 10646) character whose value is the hexadecimal value HHHH (one to four hex digits)\n",
    "\n",
    "\\UHHHHHHHH\n",
    "the Unicode (ISO/IEC 10646) character whose value is the hexadecimal value HHHHHHHH (one to eight hex digits)\n",
    "\n",
    "\\cx\n",
    "a control-x character\n",
    "\n",
    "The expanded result is single-quoted, as if the dollar sign had not been present."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "3.1.2.4 ANSI-C 인용\n",
    "$’string’ 형식의 문자 시퀀스는 특수한 종류의 작은따옴표로 처리됩니다. 이 시퀀스는 string으로 확장되며, string 안에 있는 백슬래시로 이스케이프된 문자는 ANSI C 표준에 따라 대체됩니다. 백슬래시 이스케이프 시퀀스가 있으면 다음과 같이 디코딩됩니다:\n",
    "\n",
    "\\a: 경고음(벨)\n",
    "\\b: 백스페이스\n",
    "\\e 또는 \\E: 이스케이프 문자(ANSI C가 아님)\n",
    "\\f: 폼 피드\n",
    "\\n: 줄 바꿈\n",
    "\\r: 캐리지 리턴\n",
    "\\t: 수평 탭\n",
    "\\v: 수직 탭\n",
    "\\: 백슬래시\n",
    "': 작은따옴표\n",
    "\": 큰따옴표\n",
    "?: 물음표\n",
    "\\nnn: 8비트 문자로, nnn은 1~3자리의 8진수 값\n",
    "\\xHH: 8비트 문자로, HH는 1~2자리의 16진수 값\n",
    "\\uHHHH: 유니코드(ISO/IEC 10646) 문자로, HHHH는 1~4자리의 16진수 값\n",
    "\\UHHHHHHHH: 유니코드(ISO/IEC 10646) 문자로, HHHHHHHH는 1~8자리의 16진수 값\n",
    "\\cx: 제어 문자(control-x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "3.1.2.5 Locale-Specific Translation\n",
    "Prefixing a double-quoted string with a dollar sign (‘$’), such as $\"hello, world\", will cause the string to be translated according to the current locale. The gettext infrastructure performs the lookup and translation, using the LC_MESSAGES, TEXTDOMAINDIR, and TEXTDOMAIN shell variables, as explained below. See the gettext documentation for additional details not covered here. If the current locale is C or POSIX, if there are no translations available, of if the string is not translated, the dollar sign is ignored. Since this is a form of double quoting, the string remains double-quoted by default, whether or not it is translated and replaced. If the noexpand_translation option is enabled using the shopt builtin (see The Shopt Builtin), translated strings are single-quoted instead of double-quoted.\n",
    "\n",
    "The rest of this section is a brief overview of how you use gettext to create translations for strings in a shell script named scriptname. There are more details in the gettext documentation."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\"3.1.2.5 로케일 특정 번역\n",
    "달러 기호(‘\\$’)를 큰따옴표로 감싼 문자열 앞에 붙이면(예: $\"hello, world\"), 현재 로케일에 따라 문자열이 번역됩니다. gettext 인프라가 이 작업을 수행하며, LC_MESSAGES, TEXTDOMAINDIR, TEXTDOMAIN 쉘 변수를 사용하여 번역을 조회하고 처리합니다. 추가적인 세부 사항은 gettext 문서를 참조하십시오. 현재 로케일이 C 또는 POSIX인 경우, 번역이 없는 경우, 또는 문자열이 번역되지 않는 경우, 달러 기호는 무시됩니다. 이는 큰따옴표의 한 형태이기 때문에, 번역이 이루어지든 아니든 기본적으로 문자열은 큰따옴표로 유지됩니다. shopt 내장 명령어를 사용해 noexpand_translation 옵션을 활성화하면(참고: The Shopt Builtin), 번역된 문자열은 큰따옴표 대신 작은따옴표로 감싸집니다.\n",
    "\n",
    "이 섹션의 나머지는 scriptname이라는 쉘 스크립트에서 문자열의 번역을 생성하기 위해 gettext를 사용하는 방법에 대한 간단한 개요입니다. 더 많은 세부 사항은 gettext 문서에 나와 있습니다.\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
