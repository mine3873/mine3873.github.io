---
title: "Example Post: no thumbnail image"
date: "2023-12-01"
---

# File and File System Concept
## Racall : [Address Mapping - Logical Blocks](Chapter%2011%20Mass-Storage%20Structure.md#Address%20Mapping%20-%20Logical%20Blocks)
## File Concept
- File 
	- *Logical view* of storage unit
	- *File system* (in OS) maps files to physical storage
- *File abstraction*
	- User's view : *named stream of bytes*
	- File System's view : collection of *disk blocks*
	- File system's job : translate (name, offset) to *disk blocks* :
![[Pasted image 20240603205511.png]]
## Internal File Structure
- File : Logical view
	- File is a stream of bytes
	- Each byte is addressable by its offset from beginning of file 
		- e.g.,Read $300^{th}$ byte (start at 0) of file "foo.c" $->$ read offset 299 of "foo.c"
![[Pasted image 20240603210329.png]]
![[Pasted image 20240603205656.png]]
- All disk I/O is performed in units of one *block*
	- All basic I/O functions operate in terms of block
## File Information
- File pointer
	- When process read/writes a file, system tracks last read/write location
- Disk location of file
	- Actual file location on (hard) disk
		- e.g., block number
	- How to access ? : _Sequential / Direct access_
	- How do allocate ? : _Disk allocation method_
## File Metadata
- File Attribute (= Metadata)
	- FIle name
	- Identifier : File id. usually number
	- Location : pointer to a device and location of the file
	- Size, time, data, $\dots$
- File control block (FCB)
	- Contains information of file
		- e.g., inode in Linux
### File Metadata : Directory
- Directory 
	- A special file that contains list of file names and file attributes
	- Typically forms a tree
	- For all file operations : first search directory 
		- Root directory is known to OS a priori
## File System Structure
- *File System* : data structure on disk
- A physical disk can be *partitioned* into separate partitions 
	- sometimes several physical disks can be used as one partition
- Each partition can have a *file system*, [swap space](Chapter%2010%20VirtualMemory.md#Swapping), and so on.
- Windows
	- Each volume is mounted in separate name space as "C:" or "D:".
## File Operations
- *Create / Delete, Open / Close*
	- Allocate disk space and update file system
	- e.g. fopen(), fclose() $\dots$
- *Read / Write*
	- e.g., fread(), fwrite(), $\dots$
- More details later
# File Access and Allocation Methods
## File Access Methods
### Sequential Access
![[Pasted image 20240603211259.png]]
- Information in the file is accessed / processed in order, one record(byte) after the other
	- e.g., file read, video / audio playing $\dots$
- Most common
- Method
	- read_next () : reads the next portion of the file and automatically advances the file pointer
### Direct Access
- Allow programs to read / write records rapidly, in no particular order
	- e.g., read block 14, then read block 53, and then write block 7, $\dots$
	- e.g., database access
- Method : Use relative (logical) block number
	- Relative block number : index relative to the beginning of file 
		- e.g., first relative block of file is 0, next is 1, $\dots$
	- read (n) : n is the relative block nubmer
		- read (3) : read relative block nubmer 3
![[Pasted image 20240603211528.png]]
## Disk Allocation Methods 
- How do we store a file in disk 
	- Files are stored as disk blocks 
![[Pasted image 20240603211833.png]]
- Allocation Method 
	- How to allocate disk spaces (i.e., disk blocks) for files
	- Contiguous allocation, Linked allocation, Indexed allocation
### Contiguous Allocation
- Each file occupies a set of contiguous blocks on the disk
- Simple : only $(1)$ starting location (block #) and $(2)$ length (number of blocks) are required
![[Pasted image 20240603212015.png]]
#### Pros
- Fast I/O
	- Can transfer many bytes per each disk seek/rotation
- Easily supports direct access
	- read(n) is easy
#### Cons
- [External fragmentation](Chapter%209%20Memory.md#External%20Fragmentation) (Where did we see this before)
	- The largest contiguous chunk is insufficient for a request
	- Compaction is very time consuming. Why?
		- DISK의 속도를 이용해서 Compaction을 수행 -> 비효율적
- Handling file size extension
	- grow prediction vs. internal fragmentation
		- 파일 크기 확보를 위해 빈 공간을 크게 잡을 경우, internal fragmentation 발생
### Contiguous vs. Non-contiguous ALlocation
![[Pasted image 20240603212222.png]]
### Linked Allocation (non-contiguous)
- Each file is a linked list of disk blocks 
	- Blocks have the link, i.e., next block, information
	- Blocks may be scattered anywhere on the disk
- The directory contains pointers to the first and/or last blocks of the file
![[Pasted image 20240603212356.png]]
#### Pros
- No external fragmentation
- File extension is easy - just need free blocks available
#### Disadvantages
- Efficient only for sequential access; Inefficient to support direct-access capability
	- Must always start at beginning of file and follow the links 
	- A significant number of disk head seeks
- Reliability
	- A link failure at beginning of file -> whole file can not be used
- Extra space required for the pointers
	- e.g., 512-byte block, 4-byte link : 0.78% overhead
### Linked Allocation Variation - FAT
- File-Allocation Table (FAT)
	- A section of disk at the beginning of each partition
		- Gather all the links into a single table called FAT
		- Indexed by block number
	- FAT can be stored in memory
		- Random access time is improved
	- e.g., Microsoft Windows
![[Pasted image 20240603212707.png]]
### Indexed Allocation - inode
- Indexed allocation
	- Bring all the pointers together into one location, the index block 
		- Each file has its own index block
	- Index block
		- Array of disk block addresses
		- e.g., Unix, Linux File System
![[Pasted image 20240603212809.png]]
![[Pasted image 20240603212827.png]]
- index block = 19
	- index table이 저장된 블럭이 19번에 저장되어 있음
	- page table과 유사
#### Pros
- No external fragmentation
- Supports direct access
#### Disadvantages
- Small file : needs at least 2 blocks
- Large file : one index block may not be enough
	- e.g., 512-byte index block, 4-byte address : 512/4 = 128 pointers
	- So, maximum file size can be only 128 pointers x 512 byte = 64kbyte
### Indexed Allocation - Mapping
- What if file is *too large* -> one disk block is too small for index table
1. Direct block
![[Pasted image 20240603213112.png]]
2. Single indirect block
![[Pasted image 20240603213125.png]]
3. Combined scheme (e.g., UNIX, Linux inode)
	1. Direct blocks (for small files) and Indirect blocks (for large files)
### Indexed Allocation - UNIX inode
- Inode (index node) : File metadata
![[Pasted image 20240603213344.png]]
# File System Implementation and Operations 
## open("/a/b")
![[Pasted image 20240603213434.png]]
- open("/a/b") -> root directory -> a file -> b file 을 열어라.

![[Pasted image 20240603213442.png]]
1. 파일 위치를 알기 위해 root directory=의 meta data를 불러옴 
	-  root 의 metadata는 OS가 가지고 있음
2. 각 파일의 meta data를 불러오면 b 파일의 meta data를 불러옴
	- 프로세스 A의 PCB에는 b의 meta data가 커널 상의 Open file table의 어디에 위치하는지 가리키는 table이 있음
3. read(fd)를 통해 b의 block을 memory로 copy -> buffer cache
4. buffer cache로부터 user memory로 block 메모리가 copy

![[Pasted image 20240603213450.png]]
## File System Implementation
- In-memory structure
	- For FS management and performance improvement via caching 
	- In-memory directory structure cache
		- Directory information of recently accessed directories
![[Pasted image 20240603213549.png]]

- System-wide open-file table
	- Information (FCBs) of all open files in system
	- count : number of processes that opened the file
- Per-process open-file table
	- Pointer to the appropriate entry in system-wide open-file table for each process
![[Pasted image 20240603213818.png]]
## File operations
- Create a file
	1. Application calls file system (via system call) - e.g., fopen("/usr/new.txt")
	2. File system creates new FCB - e.g., create FCB for "/usr/new.txt"
	3. Search and Load appropriate directory into memory - e.g., add "/usr/" directory in in-memory directory structure cache
	4. update directory with new file name and FCB - e.g., add FCB in "/usr/" directory
	5. write back on disk - e.g., write updated "/usr/" directory in directory structure (Disk)
![[Pasted image 20240603214042.png]]

- Open a file
	1. Open() system call - e.g., fopen("file.txt") in C
	2. First search system-wide open-file table (in memory)
	2-1. If exists - some other process has already opened the file. then, simply create a per-process open-file table entry that points to the existing system-wide open-file table entry
		-> disk로 갈 필요 없이 per-process open-file table에 pointer 추가
![[Pasted image 20240603214252.png]]
	2-2. If not exists - search directory structure (in disk) for given file name
		$(1)$ update in-memory directory structure cache
		$(2)$ Copy FCB to system-wide open-file table entry
		$(3)$ Create a per-process open-file table entry that points to the system-wide open-file table entry (same as 2-1)
		->directory 업데이트 후 디스크로부터 FCB 데이터를 copy, 그 후 per-process open-file table에 이를 가리키는 pointer를 추
1. Open returns a pointer to per-process open-file table entry - e.g., return value of \*fopen()
![[Pasted image 20240603214533.png]]

- Close a file (Close() system call)
	- Remove in-memory metadata :
		- Per-process open-file table entry is removed
		- System-wide open-file entry's open count is decremented (removed if count = 0)
## Example : File I/O
![[Pasted image 20240603214652.png]]
# Free-Space Management
- To keep track of free disk space, the system maintains a free-space list
	- Records all free disk blocks
- Bit vector (bit map)
	- Easy to get contiguous files
- Linked list (free list)
	- Cannot get contiguous space easily
	- No waste of space
## Bit Vector (Bit Map)
- Free space list is implemented as a bit map or bit vector
![[Pasted image 20240603215440.png]]