фй8
Ї
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
А
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintџџџџџџџџџ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.10.02unknown8мз4

Adam/lstm_2/lstm_cell_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*/
shared_name Adam/lstm_2/lstm_cell_2/bias/v

2Adam/lstm_2/lstm_cell_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_2/lstm_cell_2/bias/v*
_output_shapes
:(*
dtype0
А
*Adam/lstm_2/lstm_cell_2/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
(*;
shared_name,*Adam/lstm_2/lstm_cell_2/recurrent_kernel/v
Љ
>Adam/lstm_2/lstm_cell_2/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_2/lstm_cell_2/recurrent_kernel/v*
_output_shapes

:
(*
dtype0

 Adam/lstm_2/lstm_cell_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*1
shared_name" Adam/lstm_2/lstm_cell_2/kernel/v

4Adam/lstm_2/lstm_cell_2/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_2/lstm_cell_2/kernel/v*
_output_shapes

:(*
dtype0

Adam/lstm_1/lstm_cell_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*/
shared_name Adam/lstm_1/lstm_cell_1/bias/v

2Adam/lstm_1/lstm_cell_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_1/lstm_cell_1/bias/v*
_output_shapes
:P*
dtype0
А
*Adam/lstm_1/lstm_cell_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*;
shared_name,*Adam/lstm_1/lstm_cell_1/recurrent_kernel/v
Љ
>Adam/lstm_1/lstm_cell_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_1/lstm_cell_1/recurrent_kernel/v*
_output_shapes

:P*
dtype0

 Adam/lstm_1/lstm_cell_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*1
shared_name" Adam/lstm_1/lstm_cell_1/kernel/v

4Adam/lstm_1/lstm_cell_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_1/lstm_cell_1/kernel/v*
_output_shapes

:P*
dtype0

Adam/lstm/lstm_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*+
shared_nameAdam/lstm/lstm_cell/bias/v

.Adam/lstm/lstm_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/bias/v*
_output_shapes
:x*
dtype0
Ј
&Adam/lstm/lstm_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*7
shared_name(&Adam/lstm/lstm_cell/recurrent_kernel/v
Ё
:Adam/lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp&Adam/lstm/lstm_cell/recurrent_kernel/v*
_output_shapes

:x*
dtype0

Adam/lstm/lstm_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Ex*-
shared_nameAdam/lstm/lstm_cell/kernel/v

0Adam/lstm/lstm_cell/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/kernel/v*
_output_shapes

:Ex*
dtype0

Adam/identity_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/identity_output/bias/v

/Adam/identity_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/identity_output/bias/v*
_output_shapes
:*
dtype0

Adam/identity_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/identity_output/kernel/v

1Adam/identity_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/identity_output/kernel/v*
_output_shapes

:*
dtype0

Adam/class_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/class_output/bias/v

,Adam/class_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/class_output/bias/v*
_output_shapes
:*
dtype0

Adam/class_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/class_output/kernel/v

.Adam/class_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/class_output/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0

Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
: *
dtype0

Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:@ *
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
: *
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:@ *
dtype0

Adam/fingerprint/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/fingerprint/bias/v

+Adam/fingerprint/bias/v/Read/ReadVariableOpReadVariableOpAdam/fingerprint/bias/v*
_output_shapes
:@*
dtype0

Adam/fingerprint/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
@**
shared_nameAdam/fingerprint/kernel/v

-Adam/fingerprint/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fingerprint/kernel/v*
_output_shapes

:
@*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:
*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:

*
dtype0

Adam/lstm_2/lstm_cell_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*/
shared_name Adam/lstm_2/lstm_cell_2/bias/m

2Adam/lstm_2/lstm_cell_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_2/lstm_cell_2/bias/m*
_output_shapes
:(*
dtype0
А
*Adam/lstm_2/lstm_cell_2/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
(*;
shared_name,*Adam/lstm_2/lstm_cell_2/recurrent_kernel/m
Љ
>Adam/lstm_2/lstm_cell_2/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_2/lstm_cell_2/recurrent_kernel/m*
_output_shapes

:
(*
dtype0

 Adam/lstm_2/lstm_cell_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*1
shared_name" Adam/lstm_2/lstm_cell_2/kernel/m

4Adam/lstm_2/lstm_cell_2/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_2/lstm_cell_2/kernel/m*
_output_shapes

:(*
dtype0

Adam/lstm_1/lstm_cell_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*/
shared_name Adam/lstm_1/lstm_cell_1/bias/m

2Adam/lstm_1/lstm_cell_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_1/lstm_cell_1/bias/m*
_output_shapes
:P*
dtype0
А
*Adam/lstm_1/lstm_cell_1/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*;
shared_name,*Adam/lstm_1/lstm_cell_1/recurrent_kernel/m
Љ
>Adam/lstm_1/lstm_cell_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_1/lstm_cell_1/recurrent_kernel/m*
_output_shapes

:P*
dtype0

 Adam/lstm_1/lstm_cell_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*1
shared_name" Adam/lstm_1/lstm_cell_1/kernel/m

4Adam/lstm_1/lstm_cell_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_1/lstm_cell_1/kernel/m*
_output_shapes

:P*
dtype0

Adam/lstm/lstm_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*+
shared_nameAdam/lstm/lstm_cell/bias/m

.Adam/lstm/lstm_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/bias/m*
_output_shapes
:x*
dtype0
Ј
&Adam/lstm/lstm_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*7
shared_name(&Adam/lstm/lstm_cell/recurrent_kernel/m
Ё
:Adam/lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp&Adam/lstm/lstm_cell/recurrent_kernel/m*
_output_shapes

:x*
dtype0

Adam/lstm/lstm_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Ex*-
shared_nameAdam/lstm/lstm_cell/kernel/m

0Adam/lstm/lstm_cell/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/kernel/m*
_output_shapes

:Ex*
dtype0

Adam/identity_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/identity_output/bias/m

/Adam/identity_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/identity_output/bias/m*
_output_shapes
:*
dtype0

Adam/identity_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/identity_output/kernel/m

1Adam/identity_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/identity_output/kernel/m*
_output_shapes

:*
dtype0

Adam/class_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/class_output/bias/m

,Adam/class_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/class_output/bias/m*
_output_shapes
:*
dtype0

Adam/class_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/class_output/kernel/m

.Adam/class_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/class_output/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0

Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
: *
dtype0

Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:@ *
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
: *
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:@ *
dtype0

Adam/fingerprint/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/fingerprint/bias/m

+Adam/fingerprint/bias/m/Read/ReadVariableOpReadVariableOpAdam/fingerprint/bias/m*
_output_shapes
:@*
dtype0

Adam/fingerprint/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
@**
shared_nameAdam/fingerprint/kernel/m

-Adam/fingerprint/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fingerprint/kernel/m*
_output_shapes

:
@*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:
*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:

*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

lstm_2/lstm_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_namelstm_2/lstm_cell_2/bias

+lstm_2/lstm_cell_2/bias/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_2/bias*
_output_shapes
:(*
dtype0
Ђ
#lstm_2/lstm_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
(*4
shared_name%#lstm_2/lstm_cell_2/recurrent_kernel

7lstm_2/lstm_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_2/lstm_cell_2/recurrent_kernel*
_output_shapes

:
(*
dtype0

lstm_2/lstm_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(**
shared_namelstm_2/lstm_cell_2/kernel

-lstm_2/lstm_cell_2/kernel/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_2/kernel*
_output_shapes

:(*
dtype0

lstm_1/lstm_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*(
shared_namelstm_1/lstm_cell_1/bias

+lstm_1/lstm_cell_1/bias/Read/ReadVariableOpReadVariableOplstm_1/lstm_cell_1/bias*
_output_shapes
:P*
dtype0
Ђ
#lstm_1/lstm_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*4
shared_name%#lstm_1/lstm_cell_1/recurrent_kernel

7lstm_1/lstm_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_1/lstm_cell_1/recurrent_kernel*
_output_shapes

:P*
dtype0

lstm_1/lstm_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P**
shared_namelstm_1/lstm_cell_1/kernel

-lstm_1/lstm_cell_1/kernel/Read/ReadVariableOpReadVariableOplstm_1/lstm_cell_1/kernel*
_output_shapes

:P*
dtype0
~
lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*$
shared_namelstm/lstm_cell/bias
w
'lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell/bias*
_output_shapes
:x*
dtype0

lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*0
shared_name!lstm/lstm_cell/recurrent_kernel

3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/recurrent_kernel*
_output_shapes

:x*
dtype0

lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Ex*&
shared_namelstm/lstm_cell/kernel

)lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/kernel*
_output_shapes

:Ex*
dtype0

identity_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameidentity_output/bias
y
(identity_output/bias/Read/ReadVariableOpReadVariableOpidentity_output/bias*
_output_shapes
:*
dtype0

identity_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameidentity_output/kernel

*identity_output/kernel/Read/ReadVariableOpReadVariableOpidentity_output/kernel*
_output_shapes

:*
dtype0
z
class_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameclass_output/bias
s
%class_output/bias/Read/ReadVariableOpReadVariableOpclass_output/bias*
_output_shapes
:*
dtype0

class_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameclass_output/kernel
{
'class_output/kernel/Read/ReadVariableOpReadVariableOpclass_output/kernel*
_output_shapes

:*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

: *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

: *
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
: *
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:@ *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
: *
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@ *
dtype0
x
fingerprint/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_namefingerprint/bias
q
$fingerprint/bias/Read/ReadVariableOpReadVariableOpfingerprint/bias*
_output_shapes
:@*
dtype0

fingerprint/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
@*#
shared_namefingerprint/kernel
y
&fingerprint/kernel/Read/ReadVariableOpReadVariableOpfingerprint/kernel*
_output_shapes

:
@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:

*
dtype0

serving_default_input_1Placeholder*+
_output_shapes
:џџџџџџџџџE*
dtype0* 
shape:џџџџџџџџџE

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1lstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biaslstm_1/lstm_cell_1/kernel#lstm_1/lstm_cell_1/recurrent_kernellstm_1/lstm_cell_1/biaslstm_2/lstm_cell_2/kernel#lstm_2/lstm_cell_2/recurrent_kernellstm_2/lstm_cell_2/biasdense/kernel
dense/biasfingerprint/kernelfingerprint/biasdense_3/kerneldense_3/biasdense_1/kerneldense_1/biasdense_4/kerneldense_4/biasdense_2/kerneldense_2/biasidentity_output/kernelidentity_output/biasclass_output/kernelclass_output/bias*%
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_303778

NoOpNoOp
А
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*МЏ
valueБЏB­Џ BЅЏ
Ё
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures*
* 
С
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell
 
state_spec*
Ѕ
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator* 
С
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
._random_generator
/cell
0
state_spec*
С
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7_random_generator
8cell
9
state_spec*
І
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias*
І
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias*
І
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

Pkernel
Qbias*
І
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses

Xkernel
Ybias*
І
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias*
І
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias*
І
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses

pkernel
qbias*
І
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

xkernel
ybias*
Х
z0
{1
|2
}3
~4
5
6
7
8
@9
A10
H11
I12
P13
Q14
X15
Y16
`17
a18
h19
i20
p21
q22
x23
y24*
Х
z0
{1
|2
}3
~4
5
6
7
8
@9
A10
H11
I12
P13
Q14
X15
Y16
`17
a18
h19
i20
p21
q22
x23
y24*
* 
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
У
	iter
beta_1
beta_2

decay
learning_rate@mЯAmаHmбImвPmгQmдXmеYmж`mзamиhmйimкpmлqmмxmнymоzmп{mр|mс}mт~mуmф	mх	mц	mч@vшAvщHvъIvыPvьQvэXvюYvя`v№avёhvђivѓpvєqvѕxvіyvїzvј{vљ|vњ}vћ~vќv§	vў	vџ	v*
* 

serving_default* 

z0
{1
|2*

z0
{1
|2*
* 
Ѕ
states
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
 trace_0
Ёtrace_1
Ђtrace_2
Ѓtrace_3* 
* 
ы
Є	variables
Ѕtrainable_variables
Іregularization_losses
Ї	keras_api
Ј__call__
+Љ&call_and_return_all_conditional_losses
Њ_random_generator
Ћ
state_size

zkernel
{recurrent_kernel
|bias*
* 
* 
* 
* 

Ќnon_trainable_variables
­layers
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses* 

Бtrace_0
Вtrace_1* 

Гtrace_0
Дtrace_1* 
* 

}0
~1
2*

}0
~1
2*
* 
Ѕ
Еstates
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*
:
Лtrace_0
Мtrace_1
Нtrace_2
Оtrace_3* 
:
Пtrace_0
Рtrace_1
Сtrace_2
Тtrace_3* 
* 
ы
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses
Щ_random_generator
Ъ
state_size

}kernel
~recurrent_kernel
bias*
* 

0
1
2*

0
1
2*
* 
Ѕ
Ыstates
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*
:
бtrace_0
вtrace_1
гtrace_2
дtrace_3* 
:
еtrace_0
жtrace_1
зtrace_2
иtrace_3* 
* 
ю
й	variables
кtrainable_variables
лregularization_losses
м	keras_api
н__call__
+о&call_and_return_all_conditional_losses
п_random_generator
р
state_size
kernel
recurrent_kernel
	bias*
* 

@0
A1*

@0
A1*
* 

сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

цtrace_0* 

чtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

H0
I1*

H0
I1*
* 

шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

эtrace_0* 

юtrace_0* 
b\
VARIABLE_VALUEfingerprint/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEfingerprint/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

P0
Q1*

P0
Q1*
* 

яnon_trainable_variables
№layers
ёmetrics
 ђlayer_regularization_losses
ѓlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

єtrace_0* 

ѕtrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

X0
Y1*

X0
Y1*
* 

іnon_trainable_variables
їlayers
јmetrics
 љlayer_regularization_losses
њlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*

ћtrace_0* 

ќtrace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

`0
a1*

`0
a1*
* 

§non_trainable_variables
ўlayers
џmetrics
 layer_regularization_losses
layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

trace_0* 

trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

h0
i1*

h0
i1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

trace_0* 

trace_0* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

p0
q1*

p0
q1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*

trace_0* 

trace_0* 
c]
VARIABLE_VALUEclass_output/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEclass_output/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

x0
y1*

x0
y1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*

trace_0* 

trace_0* 
ga
VARIABLE_VALUEidentity_output/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEidentity_output/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUElstm/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUElstm/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUElstm/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_1/lstm_cell_1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#lstm_1/lstm_cell_1/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUElstm_1/lstm_cell_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_2/lstm_cell_2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#lstm_2/lstm_cell_2/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUElstm_2/lstm_cell_2/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
* 
b
0
1
2
3
4
5
6
7
	8

9
10
11
12*
,
0
1
2
3
4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

z0
{1
|2*

z0
{1
|2*
* 

non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
Є	variables
Ѕtrainable_variables
Іregularization_losses
Ј__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses*

Ѓtrace_0
Єtrace_1* 

Ѕtrace_0
Іtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

/0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

}0
~1
2*

}0
~1
2*
* 

Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses*

Ќtrace_0
­trace_1* 

Ўtrace_0
Џtrace_1* 
* 
* 
* 
* 

80*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1
2*

0
1
2*
* 

Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
й	variables
кtrainable_variables
лregularization_losses
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses*

Еtrace_0
Жtrace_1* 

Зtrace_0
Иtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
Й	variables
К	keras_api

Лtotal

Мcount*
<
Н	variables
О	keras_api

Пtotal

Рcount*
<
С	variables
Т	keras_api

Уtotal

Фcount*
M
Х	variables
Ц	keras_api

Чtotal

Шcount
Щ
_fn_kwargs*
M
Ъ	variables
Ы	keras_api

Ьtotal

Эcount
Ю
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Л0
М1*

Й	variables*
UO
VARIABLE_VALUEtotal_44keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_44keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

П0
Р1*

Н	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

У0
Ф1*

С	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*

Ч0
Ш1*

Х	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ь0
Э1*

Ъ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/fingerprint/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/fingerprint/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/class_output/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/class_output/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/identity_output/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/identity_output/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/lstm/lstm_cell/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/lstm/lstm_cell/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/lstm/lstm_cell/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_1/lstm_cell_1/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_1/lstm_cell_1/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_1/lstm_cell_1/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_2/lstm_cell_2/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_2/lstm_cell_2/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_2/lstm_cell_2/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/fingerprint/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/fingerprint/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/class_output/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/class_output/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/identity_output/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/identity_output/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/lstm/lstm_cell/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/lstm/lstm_cell/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/lstm/lstm_cell/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_1/lstm_cell_1/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_1/lstm_cell_1/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_1/lstm_cell_1/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_2/lstm_cell_2/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_2/lstm_cell_2/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_2/lstm_cell_2/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
!
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp&fingerprint/kernel/Read/ReadVariableOp$fingerprint/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp'class_output/kernel/Read/ReadVariableOp%class_output/bias/Read/ReadVariableOp*identity_output/kernel/Read/ReadVariableOp(identity_output/bias/Read/ReadVariableOp)lstm/lstm_cell/kernel/Read/ReadVariableOp3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp'lstm/lstm_cell/bias/Read/ReadVariableOp-lstm_1/lstm_cell_1/kernel/Read/ReadVariableOp7lstm_1/lstm_cell_1/recurrent_kernel/Read/ReadVariableOp+lstm_1/lstm_cell_1/bias/Read/ReadVariableOp-lstm_2/lstm_cell_2/kernel/Read/ReadVariableOp7lstm_2/lstm_cell_2/recurrent_kernel/Read/ReadVariableOp+lstm_2/lstm_cell_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp-Adam/fingerprint/kernel/m/Read/ReadVariableOp+Adam/fingerprint/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp.Adam/class_output/kernel/m/Read/ReadVariableOp,Adam/class_output/bias/m/Read/ReadVariableOp1Adam/identity_output/kernel/m/Read/ReadVariableOp/Adam/identity_output/bias/m/Read/ReadVariableOp0Adam/lstm/lstm_cell/kernel/m/Read/ReadVariableOp:Adam/lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOp.Adam/lstm/lstm_cell/bias/m/Read/ReadVariableOp4Adam/lstm_1/lstm_cell_1/kernel/m/Read/ReadVariableOp>Adam/lstm_1/lstm_cell_1/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_1/lstm_cell_1/bias/m/Read/ReadVariableOp4Adam/lstm_2/lstm_cell_2/kernel/m/Read/ReadVariableOp>Adam/lstm_2/lstm_cell_2/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_2/lstm_cell_2/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp-Adam/fingerprint/kernel/v/Read/ReadVariableOp+Adam/fingerprint/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp.Adam/class_output/kernel/v/Read/ReadVariableOp,Adam/class_output/bias/v/Read/ReadVariableOp1Adam/identity_output/kernel/v/Read/ReadVariableOp/Adam/identity_output/bias/v/Read/ReadVariableOp0Adam/lstm/lstm_cell/kernel/v/Read/ReadVariableOp:Adam/lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOp.Adam/lstm/lstm_cell/bias/v/Read/ReadVariableOp4Adam/lstm_1/lstm_cell_1/kernel/v/Read/ReadVariableOp>Adam/lstm_1/lstm_cell_1/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_1/lstm_cell_1/bias/v/Read/ReadVariableOp4Adam/lstm_2/lstm_cell_2/kernel/v/Read/ReadVariableOp>Adam/lstm_2/lstm_cell_2/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_2/lstm_cell_2/bias/v/Read/ReadVariableOpConst*g
Tin`
^2\	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_307492

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasfingerprint/kernelfingerprint/biasdense_1/kerneldense_1/biasdense_3/kerneldense_3/biasdense_2/kerneldense_2/biasdense_4/kerneldense_4/biasclass_output/kernelclass_output/biasidentity_output/kernelidentity_output/biaslstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biaslstm_1/lstm_cell_1/kernel#lstm_1/lstm_cell_1/recurrent_kernellstm_1/lstm_cell_1/biaslstm_2/lstm_cell_2/kernel#lstm_2/lstm_cell_2/recurrent_kernellstm_2/lstm_cell_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_4count_4total_3count_3total_2count_2total_1count_1totalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/fingerprint/kernel/mAdam/fingerprint/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/class_output/kernel/mAdam/class_output/bias/mAdam/identity_output/kernel/mAdam/identity_output/bias/mAdam/lstm/lstm_cell/kernel/m&Adam/lstm/lstm_cell/recurrent_kernel/mAdam/lstm/lstm_cell/bias/m Adam/lstm_1/lstm_cell_1/kernel/m*Adam/lstm_1/lstm_cell_1/recurrent_kernel/mAdam/lstm_1/lstm_cell_1/bias/m Adam/lstm_2/lstm_cell_2/kernel/m*Adam/lstm_2/lstm_cell_2/recurrent_kernel/mAdam/lstm_2/lstm_cell_2/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/fingerprint/kernel/vAdam/fingerprint/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/class_output/kernel/vAdam/class_output/bias/vAdam/identity_output/kernel/vAdam/identity_output/bias/vAdam/lstm/lstm_cell/kernel/v&Adam/lstm/lstm_cell/recurrent_kernel/vAdam/lstm/lstm_cell/bias/v Adam/lstm_1/lstm_cell_1/kernel/v*Adam/lstm_1/lstm_cell_1/recurrent_kernel/vAdam/lstm_1/lstm_cell_1/bias/v Adam/lstm_2/lstm_cell_2/kernel/v*Adam/lstm_2/lstm_cell_2/recurrent_kernel/vAdam/lstm_2/lstm_cell_2/bias/v*f
Tin_
]2[*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_307772ќВ1
ЂI

@__inference_lstm_layer_call_and_return_conditional_losses_305191
inputs_0:
(lstm_cell_matmul_readvariableop_resource:Ex<
*lstm_cell_matmul_1_readvariableop_resource:x7
)lstm_cell_biasadd_readvariableop_resource:x
identityЂ lstm_cell/BiasAdd/ReadVariableOpЂlstm_cell/MatMul/ReadVariableOpЂ!lstm_cell/MatMul_1/ReadVariableOpЂwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџED
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџE*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:Ex*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџx
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџq
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџr
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџj
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : љ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_305107*
condR
while_cond_305106*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџЗ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџE: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџE
"
_user_specified_name
inputs/0
Ћ
D
(__inference_dropout_layer_call_fn_305482

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_302244d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
чу
э8
"__inference__traced_restore_307772
file_prefix/
assignvariableop_dense_kernel:

+
assignvariableop_1_dense_bias:
7
%assignvariableop_2_fingerprint_kernel:
@1
#assignvariableop_3_fingerprint_bias:@3
!assignvariableop_4_dense_1_kernel:@ -
assignvariableop_5_dense_1_bias: 3
!assignvariableop_6_dense_3_kernel:@ -
assignvariableop_7_dense_3_bias: 3
!assignvariableop_8_dense_2_kernel: -
assignvariableop_9_dense_2_bias:4
"assignvariableop_10_dense_4_kernel: .
 assignvariableop_11_dense_4_bias:9
'assignvariableop_12_class_output_kernel:3
%assignvariableop_13_class_output_bias:<
*assignvariableop_14_identity_output_kernel:6
(assignvariableop_15_identity_output_bias:;
)assignvariableop_16_lstm_lstm_cell_kernel:ExE
3assignvariableop_17_lstm_lstm_cell_recurrent_kernel:x5
'assignvariableop_18_lstm_lstm_cell_bias:x?
-assignvariableop_19_lstm_1_lstm_cell_1_kernel:PI
7assignvariableop_20_lstm_1_lstm_cell_1_recurrent_kernel:P9
+assignvariableop_21_lstm_1_lstm_cell_1_bias:P?
-assignvariableop_22_lstm_2_lstm_cell_2_kernel:(I
7assignvariableop_23_lstm_2_lstm_cell_2_recurrent_kernel:
(9
+assignvariableop_24_lstm_2_lstm_cell_2_bias:('
assignvariableop_25_adam_iter:	 )
assignvariableop_26_adam_beta_1: )
assignvariableop_27_adam_beta_2: (
assignvariableop_28_adam_decay: 0
&assignvariableop_29_adam_learning_rate: %
assignvariableop_30_total_4: %
assignvariableop_31_count_4: %
assignvariableop_32_total_3: %
assignvariableop_33_count_3: %
assignvariableop_34_total_2: %
assignvariableop_35_count_2: %
assignvariableop_36_total_1: %
assignvariableop_37_count_1: #
assignvariableop_38_total: #
assignvariableop_39_count: 9
'assignvariableop_40_adam_dense_kernel_m:

3
%assignvariableop_41_adam_dense_bias_m:
?
-assignvariableop_42_adam_fingerprint_kernel_m:
@9
+assignvariableop_43_adam_fingerprint_bias_m:@;
)assignvariableop_44_adam_dense_1_kernel_m:@ 5
'assignvariableop_45_adam_dense_1_bias_m: ;
)assignvariableop_46_adam_dense_3_kernel_m:@ 5
'assignvariableop_47_adam_dense_3_bias_m: ;
)assignvariableop_48_adam_dense_2_kernel_m: 5
'assignvariableop_49_adam_dense_2_bias_m:;
)assignvariableop_50_adam_dense_4_kernel_m: 5
'assignvariableop_51_adam_dense_4_bias_m:@
.assignvariableop_52_adam_class_output_kernel_m::
,assignvariableop_53_adam_class_output_bias_m:C
1assignvariableop_54_adam_identity_output_kernel_m:=
/assignvariableop_55_adam_identity_output_bias_m:B
0assignvariableop_56_adam_lstm_lstm_cell_kernel_m:ExL
:assignvariableop_57_adam_lstm_lstm_cell_recurrent_kernel_m:x<
.assignvariableop_58_adam_lstm_lstm_cell_bias_m:xF
4assignvariableop_59_adam_lstm_1_lstm_cell_1_kernel_m:PP
>assignvariableop_60_adam_lstm_1_lstm_cell_1_recurrent_kernel_m:P@
2assignvariableop_61_adam_lstm_1_lstm_cell_1_bias_m:PF
4assignvariableop_62_adam_lstm_2_lstm_cell_2_kernel_m:(P
>assignvariableop_63_adam_lstm_2_lstm_cell_2_recurrent_kernel_m:
(@
2assignvariableop_64_adam_lstm_2_lstm_cell_2_bias_m:(9
'assignvariableop_65_adam_dense_kernel_v:

3
%assignvariableop_66_adam_dense_bias_v:
?
-assignvariableop_67_adam_fingerprint_kernel_v:
@9
+assignvariableop_68_adam_fingerprint_bias_v:@;
)assignvariableop_69_adam_dense_1_kernel_v:@ 5
'assignvariableop_70_adam_dense_1_bias_v: ;
)assignvariableop_71_adam_dense_3_kernel_v:@ 5
'assignvariableop_72_adam_dense_3_bias_v: ;
)assignvariableop_73_adam_dense_2_kernel_v: 5
'assignvariableop_74_adam_dense_2_bias_v:;
)assignvariableop_75_adam_dense_4_kernel_v: 5
'assignvariableop_76_adam_dense_4_bias_v:@
.assignvariableop_77_adam_class_output_kernel_v::
,assignvariableop_78_adam_class_output_bias_v:C
1assignvariableop_79_adam_identity_output_kernel_v:=
/assignvariableop_80_adam_identity_output_bias_v:B
0assignvariableop_81_adam_lstm_lstm_cell_kernel_v:ExL
:assignvariableop_82_adam_lstm_lstm_cell_recurrent_kernel_v:x<
.assignvariableop_83_adam_lstm_lstm_cell_bias_v:xF
4assignvariableop_84_adam_lstm_1_lstm_cell_1_kernel_v:PP
>assignvariableop_85_adam_lstm_1_lstm_cell_1_recurrent_kernel_v:P@
2assignvariableop_86_adam_lstm_1_lstm_cell_1_bias_v:PF
4assignvariableop_87_adam_lstm_2_lstm_cell_2_kernel_v:(P
>assignvariableop_88_adam_lstm_2_lstm_cell_2_recurrent_kernel_v:
(@
2assignvariableop_89_adam_lstm_2_lstm_cell_2_bias_v:(
identity_91ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_70ЂAssignVariableOp_71ЂAssignVariableOp_72ЂAssignVariableOp_73ЂAssignVariableOp_74ЂAssignVariableOp_75ЂAssignVariableOp_76ЂAssignVariableOp_77ЂAssignVariableOp_78ЂAssignVariableOp_79ЂAssignVariableOp_8ЂAssignVariableOp_80ЂAssignVariableOp_81ЂAssignVariableOp_82ЂAssignVariableOp_83ЂAssignVariableOp_84ЂAssignVariableOp_85ЂAssignVariableOp_86ЂAssignVariableOp_87ЂAssignVariableOp_88ЂAssignVariableOp_89ЂAssignVariableOp_9/
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:[*
dtype0*К.
valueА.B­.[B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЉ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:[*
dtype0*Ы
valueСBО[B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ш
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesя
ь:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*i
dtypes_
]2[	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp%assignvariableop_2_fingerprint_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp#assignvariableop_3_fingerprint_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp'assignvariableop_12_class_output_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp%assignvariableop_13_class_output_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp*assignvariableop_14_identity_output_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp(assignvariableop_15_identity_output_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp)assignvariableop_16_lstm_lstm_cell_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_17AssignVariableOp3assignvariableop_17_lstm_lstm_cell_recurrent_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp'assignvariableop_18_lstm_lstm_cell_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp-assignvariableop_19_lstm_1_lstm_cell_1_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_20AssignVariableOp7assignvariableop_20_lstm_1_lstm_cell_1_recurrent_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp+assignvariableop_21_lstm_1_lstm_cell_1_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp-assignvariableop_22_lstm_2_lstm_cell_2_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_23AssignVariableOp7assignvariableop_23_lstm_2_lstm_cell_2_recurrent_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp+assignvariableop_24_lstm_2_lstm_cell_2_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_iterIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_beta_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_beta_2Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_decayIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp&assignvariableop_29_adam_learning_rateIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOpassignvariableop_30_total_4Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOpassignvariableop_31_count_4Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOpassignvariableop_32_total_3Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOpassignvariableop_33_count_3Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOpassignvariableop_34_total_2Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOpassignvariableop_35_count_2Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOpassignvariableop_36_total_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOpassignvariableop_37_count_1Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOpassignvariableop_38_totalIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOpassignvariableop_39_countIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp%assignvariableop_41_adam_dense_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp-assignvariableop_42_adam_fingerprint_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_fingerprint_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_dense_1_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_dense_1_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_dense_3_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp'assignvariableop_47_adam_dense_3_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_dense_2_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp'assignvariableop_49_adam_dense_2_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_dense_4_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp'assignvariableop_51_adam_dense_4_bias_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp.assignvariableop_52_adam_class_output_kernel_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp,assignvariableop_53_adam_class_output_bias_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_54AssignVariableOp1assignvariableop_54_adam_identity_output_kernel_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_55AssignVariableOp/assignvariableop_55_adam_identity_output_bias_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_56AssignVariableOp0assignvariableop_56_adam_lstm_lstm_cell_kernel_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_57AssignVariableOp:assignvariableop_57_adam_lstm_lstm_cell_recurrent_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp.assignvariableop_58_adam_lstm_lstm_cell_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_59AssignVariableOp4assignvariableop_59_adam_lstm_1_lstm_cell_1_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_60AssignVariableOp>assignvariableop_60_adam_lstm_1_lstm_cell_1_recurrent_kernel_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_61AssignVariableOp2assignvariableop_61_adam_lstm_1_lstm_cell_1_bias_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_62AssignVariableOp4assignvariableop_62_adam_lstm_2_lstm_cell_2_kernel_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_63AssignVariableOp>assignvariableop_63_adam_lstm_2_lstm_cell_2_recurrent_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_64AssignVariableOp2assignvariableop_64_adam_lstm_2_lstm_cell_2_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_65AssignVariableOp'assignvariableop_65_adam_dense_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_66AssignVariableOp%assignvariableop_66_adam_dense_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_67AssignVariableOp-assignvariableop_67_adam_fingerprint_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_68AssignVariableOp+assignvariableop_68_adam_fingerprint_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp)assignvariableop_69_adam_dense_1_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp'assignvariableop_70_adam_dense_1_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOp)assignvariableop_71_adam_dense_3_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_72AssignVariableOp'assignvariableop_72_adam_dense_3_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_73AssignVariableOp)assignvariableop_73_adam_dense_2_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_74AssignVariableOp'assignvariableop_74_adam_dense_2_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_75AssignVariableOp)assignvariableop_75_adam_dense_4_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_76AssignVariableOp'assignvariableop_76_adam_dense_4_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_77AssignVariableOp.assignvariableop_77_adam_class_output_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_78AssignVariableOp,assignvariableop_78_adam_class_output_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_79AssignVariableOp1assignvariableop_79_adam_identity_output_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_80AssignVariableOp/assignvariableop_80_adam_identity_output_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_81AssignVariableOp0assignvariableop_81_adam_lstm_lstm_cell_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_82AssignVariableOp:assignvariableop_82_adam_lstm_lstm_cell_recurrent_kernel_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_83AssignVariableOp.assignvariableop_83_adam_lstm_lstm_cell_bias_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_84AssignVariableOp4assignvariableop_84_adam_lstm_1_lstm_cell_1_kernel_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_85AssignVariableOp>assignvariableop_85_adam_lstm_1_lstm_cell_1_recurrent_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_86AssignVariableOp2assignvariableop_86_adam_lstm_1_lstm_cell_1_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_87AssignVariableOp4assignvariableop_87_adam_lstm_2_lstm_cell_2_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_88AssignVariableOp>assignvariableop_88_adam_lstm_2_lstm_cell_2_recurrent_kernel_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_89AssignVariableOp2assignvariableop_89_adam_lstm_2_lstm_cell_2_bias_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_90Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_91IdentityIdentity_90:output:0^NoOp_1*
T0*
_output_shapes
: ј
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_91Identity_91:output:0*Ы
_input_shapesЙ
Ж: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Г6
Џ
while_body_305107
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:ExD
2while_lstm_cell_matmul_1_readvariableop_resource_0:x?
1while_lstm_cell_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:ExB
0while_lstm_cell_matmul_1_readvariableop_resource:x=
/while_lstm_cell_biasadd_readvariableop_resource:xЂ&while/lstm_cell/BiasAdd/ReadVariableOpЂ%while/lstm_cell/MatMul/ReadVariableOpЂ'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџE*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:Ex*
dtype0Г
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџx
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџv
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџn
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџv
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџk
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџТ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџv
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџЧ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
С	
Ї
lstm_while_cond_304431&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1>
:lstm_while_lstm_while_cond_304431___redundant_placeholder0>
:lstm_while_lstm_while_cond_304431___redundant_placeholder1>
:lstm_while_lstm_while_cond_304431___redundant_placeholder2>
:lstm_while_lstm_while_cond_304431___redundant_placeholder3
lstm_while_identity
v
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: U
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: "3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:


ђ
A__inference_dense_layer_call_and_return_conditional_losses_302559

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
і
Џ
%__inference_lstm_layer_call_fn_304905

inputs
unknown:Ex
	unknown_0:x
	unknown_1:x
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_303332s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџE: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs
љ	
Я
lstm_1_while_cond_304578*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3,
(lstm_1_while_less_lstm_1_strided_slice_1B
>lstm_1_while_lstm_1_while_cond_304578___redundant_placeholder0B
>lstm_1_while_lstm_1_while_cond_304578___redundant_placeholder1B
>lstm_1_while_lstm_1_while_cond_304578___redundant_placeholder2B
>lstm_1_while_lstm_1_while_cond_304578___redundant_placeholder3
lstm_1_while_identity
~
lstm_1/while/LessLesslstm_1_while_placeholder(lstm_1_while_less_lstm_1_strided_slice_1*
T0*
_output_shapes
: Y
lstm_1/while/IdentityIdentitylstm_1/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_1_while_identitylstm_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Е
У
while_cond_302303
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_302303___redundant_placeholder04
0while_while_cond_302303___redundant_placeholder14
0while_while_cond_302303___redundant_placeholder24
0while_while_cond_302303___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
С	
Ї
lstm_while_cond_303950&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1>
:lstm_while_lstm_while_cond_303950___redundant_placeholder0>
:lstm_while_lstm_while_cond_303950___redundant_placeholder1>
:lstm_while_lstm_while_cond_303950___redundant_placeholder2>
:lstm_while_lstm_while_cond_303950___redundant_placeholder3
lstm_while_identity
v
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: U
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: "3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Е
У
while_cond_301458
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_301458___redundant_placeholder04
0while_while_cond_301458___redundant_placeholder14
0while_while_cond_301458___redundant_placeholder24
0while_while_cond_301458___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
п7
ѓ
@__inference_lstm_layer_call_and_return_conditional_losses_301369

inputs"
lstm_cell_301287:Ex"
lstm_cell_301289:x
lstm_cell_301291:x
identityЂ!lstm_cell/StatefulPartitionedCallЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџED
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџE*
shrink_axis_maskц
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_301287lstm_cell_301289lstm_cell_301291*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_301241n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ў
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_301287lstm_cell_301289lstm_cell_301291*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_301300*
condR
while_cond_301299*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџr
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџE: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџE
 
_user_specified_nameinputs
"
Ш
while_body_301109
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_301133_0:Ex*
while_lstm_cell_301135_0:x&
while_lstm_cell_301137_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_301133:Ex(
while_lstm_cell_301135:x$
while_lstm_cell_301137:xЂ'while/lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџE*
element_dtype0Є
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_301133_0while_lstm_cell_301135_0while_lstm_cell_301137_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_301095й
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџv

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"2
while_lstm_cell_301133while_lstm_cell_301133_0"2
while_lstm_cell_301135while_lstm_cell_301135_0"2
while_lstm_cell_301137while_lstm_cell_301137_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
а

0__inference_identity_output_layer_call_fn_306893

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_identity_output_layer_call_and_return_conditional_losses_302661o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
њ
Б
'__inference_lstm_1_layer_call_fn_305537

inputs
unknown:P
	unknown_0:P
	unknown_1:P
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_302388s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
J

B__inference_lstm_1_layer_call_and_return_conditional_losses_305834
inputs_0<
*lstm_cell_1_matmul_readvariableop_resource:P>
,lstm_cell_1_matmul_1_readvariableop_resource:P9
+lstm_cell_1_biasadd_readvariableop_resource:P
identityЂ"lstm_cell_1/BiasAdd/ReadVariableOpЂ!lstm_cell_1/MatMul/ReadVariableOpЂ#lstm_cell_1/MatMul_1/ReadVariableOpЂwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype0
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :к
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitl
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџn
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџu
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
lstm_cell_1/ReluRelulstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџx
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџc
lstm_cell_1/Relu_1Relulstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0 lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_1_matmul_readvariableop_resource,lstm_cell_1_matmul_1_readvariableop_resource+lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_305750*
condR
while_cond_305749*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџН
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
М

&__inference_dense_layer_call_fn_306753

inputs
unknown:


	unknown_0:

identityЂStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_302559o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
Г6
Џ
while_body_305393
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:ExD
2while_lstm_cell_matmul_1_readvariableop_resource_0:x?
1while_lstm_cell_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:ExB
0while_lstm_cell_matmul_1_readvariableop_resource:x=
/while_lstm_cell_biasadd_readvariableop_resource:xЂ&while/lstm_cell/BiasAdd/ReadVariableOpЂ%while/lstm_cell/MatMul/ReadVariableOpЂ'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџE*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:Ex*
dtype0Г
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџx
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџv
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџn
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџv
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџk
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџТ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџv
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџЧ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
?
Ш

A__inference_model_layer_call_and_return_conditional_losses_303713
input_1
lstm_303649:Ex
lstm_303651:x
lstm_303653:x
lstm_1_303657:P
lstm_1_303659:P
lstm_1_303661:P
lstm_2_303664:(
lstm_2_303666:
(
lstm_2_303668:(
dense_303671:


dense_303673:
$
fingerprint_303676:
@ 
fingerprint_303678:@ 
dense_3_303681:@ 
dense_3_303683:  
dense_1_303686:@ 
dense_1_303688:  
dense_4_303691: 
dense_4_303693: 
dense_2_303696: 
dense_2_303698:(
identity_output_303701:$
identity_output_303703:%
class_output_303706:!
class_output_303708:
identity

identity_1Ђ$class_output/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ#fingerprint/StatefulPartitionedCallЂ'identity_output/StatefulPartitionedCallЂlstm/StatefulPartitionedCallЂlstm_1/StatefulPartitionedCallЂlstm_2/StatefulPartitionedCallє
lstm/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_303649lstm_303651lstm_303653*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_303332щ
dropout/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_303173
lstm_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0lstm_1_303657lstm_1_303659lstm_1_303661*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_303144
lstm_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0lstm_2_303664lstm_2_303666lstm_2_303668*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_302979
dense/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_303671dense_303673*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_302559
#fingerprint/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0fingerprint_303676fingerprint_303678*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_fingerprint_layer_call_and_return_conditional_losses_302576
dense_3/StatefulPartitionedCallStatefulPartitionedCall,fingerprint/StatefulPartitionedCall:output:0dense_3_303681dense_3_303683*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_302593
dense_1/StatefulPartitionedCallStatefulPartitionedCall,fingerprint/StatefulPartitionedCall:output:0dense_1_303686dense_1_303688*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_302610
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_303691dense_4_303693*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_302627
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_303696dense_2_303698*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_302644Ў
'identity_output/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0identity_output_303701identity_output_303703*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_identity_output_layer_call_and_return_conditional_losses_302661Ђ
$class_output/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0class_output_303706class_output_303708*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_class_output_layer_call_and_return_conditional_losses_302678|
IdentityIdentity-class_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_1Identity0identity_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџш
NoOpNoOp%^class_output/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall$^fingerprint/StatefulPartitionedCall(^identity_output/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџE: : : : : : : : : : : : : : : : : : : : : : : : : 2L
$class_output/StatefulPartitionedCall$class_output/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2J
#fingerprint/StatefulPartitionedCall#fingerprint/StatefulPartitionedCall2R
'identity_output/StatefulPartitionedCall'identity_output/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџE
!
_user_specified_name	input_1
Е
У
while_cond_306658
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_306658___redundant_placeholder04
0while_while_cond_306658___redundant_placeholder14
0while_while_cond_306658___redundant_placeholder24
0while_while_cond_306658___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ
:џџџџџџџџџ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
 
Б
%__inference_lstm_layer_call_fn_304872
inputs_0
unknown:Ex
	unknown_0:x
	unknown_1:x
identityЂStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_301178|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџE: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџE
"
_user_specified_name
inputs/0
Г6
Џ
while_body_302147
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:ExD
2while_lstm_cell_matmul_1_readvariableop_resource_0:x?
1while_lstm_cell_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:ExB
0while_lstm_cell_matmul_1_readvariableop_resource:x=
/while_lstm_cell_biasadd_readvariableop_resource:xЂ&while/lstm_cell/BiasAdd/ReadVariableOpЂ%while/lstm_cell/MatMul/ReadVariableOpЂ'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџE*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:Ex*
dtype0Г
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџx
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџv
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџn
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџv
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџk
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџТ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџv
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџЧ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Л7
С
while_body_305607
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_1_matmul_readvariableop_resource_0:PF
4while_lstm_cell_1_matmul_1_readvariableop_resource_0:PA
3while_lstm_cell_1_biasadd_readvariableop_resource_0:P
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_1_matmul_readvariableop_resource:PD
2while_lstm_cell_1_matmul_1_readvariableop_resource:P?
1while_lstm_cell_1_biasadd_readvariableop_resource:PЂ(while/lstm_cell_1/BiasAdd/ReadVariableOpЂ'while/lstm_cell_1/MatMul/ReadVariableOpЂ)while/lstm_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
'while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0З
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
)while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype0
while/lstm_cell_1/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
while/lstm_cell_1/addAddV2"while/lstm_cell_1/MatMul:product:0$while/lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP
(while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0Ѓ
while/lstm_cell_1/BiasAddBiasAddwhile/lstm_cell_1/add:z:00while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPc
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ь
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0"while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitx
while/lstm_cell_1/SigmoidSigmoid while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџz
while/lstm_cell_1/Sigmoid_1Sigmoid while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mulMulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџr
while/lstm_cell_1/ReluRelu while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mul_1Mulwhile/lstm_cell_1/Sigmoid:y:0$while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/add_1AddV2while/lstm_cell_1/mul:z:0while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџz
while/lstm_cell_1/Sigmoid_2Sigmoid while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџo
while/lstm_cell_1/Relu_1Reluwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mul_2Mulwhile/lstm_cell_1/Sigmoid_2:y:0&while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_1/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџx
while/Identity_5Identitywhile/lstm_cell_1/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџЭ

while/NoOpNoOp)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_1_biasadd_readvariableop_resource3while_lstm_cell_1_biasadd_readvariableop_resource_0"j
2while_lstm_cell_1_matmul_1_readvariableop_resource4while_lstm_cell_1_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_1_matmul_readvariableop_resource2while_lstm_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2T
(while/lstm_cell_1/BiasAdd/ReadVariableOp(while/lstm_cell_1/BiasAdd/ReadVariableOp2R
'while/lstm_cell_1/MatMul/ReadVariableOp'while/lstm_cell_1/MatMul/ReadVariableOp2V
)while/lstm_cell_1/MatMul_1/ReadVariableOp)while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Л7
С
while_body_303060
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_1_matmul_readvariableop_resource_0:PF
4while_lstm_cell_1_matmul_1_readvariableop_resource_0:PA
3while_lstm_cell_1_biasadd_readvariableop_resource_0:P
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_1_matmul_readvariableop_resource:PD
2while_lstm_cell_1_matmul_1_readvariableop_resource:P?
1while_lstm_cell_1_biasadd_readvariableop_resource:PЂ(while/lstm_cell_1/BiasAdd/ReadVariableOpЂ'while/lstm_cell_1/MatMul/ReadVariableOpЂ)while/lstm_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
'while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0З
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
)while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype0
while/lstm_cell_1/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
while/lstm_cell_1/addAddV2"while/lstm_cell_1/MatMul:product:0$while/lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP
(while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0Ѓ
while/lstm_cell_1/BiasAddBiasAddwhile/lstm_cell_1/add:z:00while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPc
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ь
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0"while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitx
while/lstm_cell_1/SigmoidSigmoid while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџz
while/lstm_cell_1/Sigmoid_1Sigmoid while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mulMulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџr
while/lstm_cell_1/ReluRelu while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mul_1Mulwhile/lstm_cell_1/Sigmoid:y:0$while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/add_1AddV2while/lstm_cell_1/mul:z:0while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџz
while/lstm_cell_1/Sigmoid_2Sigmoid while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџo
while/lstm_cell_1/Relu_1Reluwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mul_2Mulwhile/lstm_cell_1/Sigmoid_2:y:0&while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_1/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџx
while/Identity_5Identitywhile/lstm_cell_1/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџЭ

while/NoOpNoOp)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_1_biasadd_readvariableop_resource3while_lstm_cell_1_biasadd_readvariableop_resource_0"j
2while_lstm_cell_1_matmul_1_readvariableop_resource4while_lstm_cell_1_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_1_matmul_readvariableop_resource2while_lstm_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2T
(while/lstm_cell_1/BiasAdd/ReadVariableOp(while/lstm_cell_1/BiasAdd/ReadVariableOp2R
'while/lstm_cell_1/MatMul/ReadVariableOp'while/lstm_cell_1/MatMul/ReadVariableOp2V
)while/lstm_cell_1/MatMul_1/ReadVariableOp)while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Ѕ<
Я	
lstm_while_body_303951&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0G
5lstm_while_lstm_cell_matmul_readvariableop_resource_0:ExI
7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:xD
6lstm_while_lstm_cell_biasadd_readvariableop_resource_0:x
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorE
3lstm_while_lstm_cell_matmul_readvariableop_resource:ExG
5lstm_while_lstm_cell_matmul_1_readvariableop_resource:xB
4lstm_while_lstm_cell_biasadd_readvariableop_resource:xЂ+lstm/while/lstm_cell/BiasAdd/ReadVariableOpЂ*lstm/while/lstm_cell/MatMul/ReadVariableOpЂ,lstm/while/lstm_cell/MatMul_1/ReadVariableOp
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   П
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџE*
element_dtype0 
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:Ex*
dtype0Т
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxЄ
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0Љ
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxЃ
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџx
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0Ќ
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxf
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ѕ
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split~
lstm/while/lstm_cell/SigmoidSigmoid#lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/while/lstm_cell/Sigmoid_1Sigmoid#lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/while/lstm_cell/mulMul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџx
lstm/while/lstm_cell/ReluRelu#lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/while/lstm_cell/mul_1Mul lstm/while/lstm_cell/Sigmoid:y:0'lstm/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/while/lstm_cell/Sigmoid_2Sigmoid#lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџu
lstm/while/lstm_cell/Relu_1Relulstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
lstm/while/lstm_cell/mul_2Mul"lstm/while/lstm_cell/Sigmoid_2:y:0)lstm/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџж
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвR
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: T
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: h
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ~
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: h
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџл
lstm/while/NoOpNoOp,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"n
4lstm_while_lstm_cell_biasadd_readvariableop_resource6lstm_while_lstm_cell_biasadd_readvariableop_resource_0"p
5lstm_while_lstm_cell_matmul_1_readvariableop_resource7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"l
3lstm_while_lstm_cell_matmul_readvariableop_resource5lstm_while_lstm_cell_matmul_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"М
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2Z
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp+lstm/while/lstm_cell/BiasAdd/ReadVariableOp2X
*lstm/while/lstm_cell/MatMul/ReadVariableOp*lstm/while/lstm_cell/MatMul/ReadVariableOp2\
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp,lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
ч=
І

A__inference_model_layer_call_and_return_conditional_losses_303646
input_1
lstm_303582:Ex
lstm_303584:x
lstm_303586:x
lstm_1_303590:P
lstm_1_303592:P
lstm_1_303594:P
lstm_2_303597:(
lstm_2_303599:
(
lstm_2_303601:(
dense_303604:


dense_303606:
$
fingerprint_303609:
@ 
fingerprint_303611:@ 
dense_3_303614:@ 
dense_3_303616:  
dense_1_303619:@ 
dense_1_303621:  
dense_4_303624: 
dense_4_303626: 
dense_2_303629: 
dense_2_303631:(
identity_output_303634:$
identity_output_303636:%
class_output_303639:!
class_output_303641:
identity

identity_1Ђ$class_output/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂ#fingerprint/StatefulPartitionedCallЂ'identity_output/StatefulPartitionedCallЂlstm/StatefulPartitionedCallЂlstm_1/StatefulPartitionedCallЂlstm_2/StatefulPartitionedCallє
lstm/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_303582lstm_303584lstm_303586*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_302231й
dropout/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_302244
lstm_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0lstm_1_303590lstm_1_303592lstm_1_303594*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_302388
lstm_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0lstm_2_303597lstm_2_303599lstm_2_303601*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_302540
dense/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_303604dense_303606*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_302559
#fingerprint/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0fingerprint_303609fingerprint_303611*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_fingerprint_layer_call_and_return_conditional_losses_302576
dense_3/StatefulPartitionedCallStatefulPartitionedCall,fingerprint/StatefulPartitionedCall:output:0dense_3_303614dense_3_303616*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_302593
dense_1/StatefulPartitionedCallStatefulPartitionedCall,fingerprint/StatefulPartitionedCall:output:0dense_1_303619dense_1_303621*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_302610
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_303624dense_4_303626*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_302627
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_303629dense_2_303631*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_302644Ў
'identity_output/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0identity_output_303634identity_output_303636*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_identity_output_layer_call_and_return_conditional_losses_302661Ђ
$class_output/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0class_output_303639class_output_303641*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_class_output_layer_call_and_return_conditional_losses_302678|
IdentityIdentity-class_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_1Identity0identity_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЦ
NoOpNoOp%^class_output/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall$^fingerprint/StatefulPartitionedCall(^identity_output/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџE: : : : : : : : : : : : : : : : : : : : : : : : : 2L
$class_output/StatefulPartitionedCall$class_output/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2J
#fingerprint/StatefulPartitionedCall#fingerprint/StatefulPartitionedCall2R
'identity_output/StatefulPartitionedCall'identity_output/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџE
!
_user_specified_name	input_1
ъ8
§
B__inference_lstm_2_layer_call_and_return_conditional_losses_301880

inputs$
lstm_cell_2_301796:($
lstm_cell_2_301798:
( 
lstm_cell_2_301800:(
identityЂ#lstm_cell_2/StatefulPartitionedCallЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask№
#lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_2_301796lstm_cell_2_301798lstm_cell_2_301800*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_301795n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Д
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_2_301796lstm_cell_2_301798lstm_cell_2_301800*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_301810*
condR
while_cond_301809*K
output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
t
NoOpNoOp$^lstm_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2J
#lstm_cell_2/StatefulPartitionedCall#lstm_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Є

љ
H__inference_class_output_layer_call_and_return_conditional_losses_306884

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

ќ
K__inference_identity_output_layer_call_and_return_conditional_losses_306904

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Е
У
while_cond_301649
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_301649___redundant_placeholder04
0while_while_cond_301649___redundant_placeholder14
0while_while_cond_301649___redundant_placeholder24
0while_while_cond_301649___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
?
Ч

A__inference_model_layer_call_and_return_conditional_losses_303467

inputs
lstm_303403:Ex
lstm_303405:x
lstm_303407:x
lstm_1_303411:P
lstm_1_303413:P
lstm_1_303415:P
lstm_2_303418:(
lstm_2_303420:
(
lstm_2_303422:(
dense_303425:


dense_303427:
$
fingerprint_303430:
@ 
fingerprint_303432:@ 
dense_3_303435:@ 
dense_3_303437:  
dense_1_303440:@ 
dense_1_303442:  
dense_4_303445: 
dense_4_303447: 
dense_2_303450: 
dense_2_303452:(
identity_output_303455:$
identity_output_303457:%
class_output_303460:!
class_output_303462:
identity

identity_1Ђ$class_output/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ#fingerprint/StatefulPartitionedCallЂ'identity_output/StatefulPartitionedCallЂlstm/StatefulPartitionedCallЂlstm_1/StatefulPartitionedCallЂlstm_2/StatefulPartitionedCallѓ
lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm_303403lstm_303405lstm_303407*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_303332щ
dropout/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_303173
lstm_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0lstm_1_303411lstm_1_303413lstm_1_303415*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_303144
lstm_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0lstm_2_303418lstm_2_303420lstm_2_303422*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_302979
dense/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_303425dense_303427*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_302559
#fingerprint/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0fingerprint_303430fingerprint_303432*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_fingerprint_layer_call_and_return_conditional_losses_302576
dense_3/StatefulPartitionedCallStatefulPartitionedCall,fingerprint/StatefulPartitionedCall:output:0dense_3_303435dense_3_303437*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_302593
dense_1/StatefulPartitionedCallStatefulPartitionedCall,fingerprint/StatefulPartitionedCall:output:0dense_1_303440dense_1_303442*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_302610
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_303445dense_4_303447*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_302627
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_303450dense_2_303452*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_302644Ў
'identity_output/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0identity_output_303455identity_output_303457*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_identity_output_layer_call_and_return_conditional_losses_302661Ђ
$class_output/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0class_output_303460class_output_303462*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_class_output_layer_call_and_return_conditional_losses_302678|
IdentityIdentity-class_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_1Identity0identity_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџш
NoOpNoOp%^class_output/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall$^fingerprint/StatefulPartitionedCall(^identity_output/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџE: : : : : : : : : : : : : : : : : : : : : : : : : 2L
$class_output/StatefulPartitionedCall$class_output/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2J
#fingerprint/StatefulPartitionedCall#fingerprint/StatefulPartitionedCall2R
'identity_output/StatefulPartitionedCall'identity_output/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs
щ
Ѕ
&__inference_model_layer_call_fn_302741
input_1
unknown:Ex
	unknown_0:x
	unknown_1:x
	unknown_2:P
	unknown_3:P
	unknown_4:P
	unknown_5:(
	unknown_6:
(
	unknown_7:(
	unknown_8:


	unknown_9:


unknown_10:
@

unknown_11:@

unknown_12:@ 

unknown_13: 

unknown_14:@ 

unknown_15: 

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:
identity

identity_1ЂStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_302686o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџE: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџE
!
_user_specified_name	input_1
K

B__inference_lstm_2_layer_call_and_return_conditional_losses_306309
inputs_0<
*lstm_cell_2_matmul_readvariableop_resource:(>
,lstm_cell_2_matmul_1_readvariableop_resource:
(9
+lstm_cell_2_biasadd_readvariableop_resource:(
identityЂ"lstm_cell_2/BiasAdd/ReadVariableOpЂ!lstm_cell_2/MatMul/ReadVariableOpЂ#lstm_cell_2/MatMul_1/ReadVariableOpЂwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype0
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :к
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitl
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
u
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
f
lstm_cell_2/ReluRelulstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
x
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
c
lstm_cell_2/Relu_1Relulstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0 lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_306224*
condR
while_cond_306223*K
output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Н
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Їб
ф
!__inference__wrapped_model_301028
input_1E
3model_lstm_lstm_cell_matmul_readvariableop_resource:ExG
5model_lstm_lstm_cell_matmul_1_readvariableop_resource:xB
4model_lstm_lstm_cell_biasadd_readvariableop_resource:xI
7model_lstm_1_lstm_cell_1_matmul_readvariableop_resource:PK
9model_lstm_1_lstm_cell_1_matmul_1_readvariableop_resource:PF
8model_lstm_1_lstm_cell_1_biasadd_readvariableop_resource:PI
7model_lstm_2_lstm_cell_2_matmul_readvariableop_resource:(K
9model_lstm_2_lstm_cell_2_matmul_1_readvariableop_resource:
(F
8model_lstm_2_lstm_cell_2_biasadd_readvariableop_resource:(<
*model_dense_matmul_readvariableop_resource:

9
+model_dense_biasadd_readvariableop_resource:
B
0model_fingerprint_matmul_readvariableop_resource:
@?
1model_fingerprint_biasadd_readvariableop_resource:@>
,model_dense_3_matmul_readvariableop_resource:@ ;
-model_dense_3_biasadd_readvariableop_resource: >
,model_dense_1_matmul_readvariableop_resource:@ ;
-model_dense_1_biasadd_readvariableop_resource: >
,model_dense_4_matmul_readvariableop_resource: ;
-model_dense_4_biasadd_readvariableop_resource:>
,model_dense_2_matmul_readvariableop_resource: ;
-model_dense_2_biasadd_readvariableop_resource:F
4model_identity_output_matmul_readvariableop_resource:C
5model_identity_output_biasadd_readvariableop_resource:C
1model_class_output_matmul_readvariableop_resource:@
2model_class_output_biasadd_readvariableop_resource:
identity

identity_1Ђ)model/class_output/BiasAdd/ReadVariableOpЂ(model/class_output/MatMul/ReadVariableOpЂ"model/dense/BiasAdd/ReadVariableOpЂ!model/dense/MatMul/ReadVariableOpЂ$model/dense_1/BiasAdd/ReadVariableOpЂ#model/dense_1/MatMul/ReadVariableOpЂ$model/dense_2/BiasAdd/ReadVariableOpЂ#model/dense_2/MatMul/ReadVariableOpЂ$model/dense_3/BiasAdd/ReadVariableOpЂ#model/dense_3/MatMul/ReadVariableOpЂ$model/dense_4/BiasAdd/ReadVariableOpЂ#model/dense_4/MatMul/ReadVariableOpЂ(model/fingerprint/BiasAdd/ReadVariableOpЂ'model/fingerprint/MatMul/ReadVariableOpЂ,model/identity_output/BiasAdd/ReadVariableOpЂ+model/identity_output/MatMul/ReadVariableOpЂ+model/lstm/lstm_cell/BiasAdd/ReadVariableOpЂ*model/lstm/lstm_cell/MatMul/ReadVariableOpЂ,model/lstm/lstm_cell/MatMul_1/ReadVariableOpЂmodel/lstm/whileЂ/model/lstm_1/lstm_cell_1/BiasAdd/ReadVariableOpЂ.model/lstm_1/lstm_cell_1/MatMul/ReadVariableOpЂ0model/lstm_1/lstm_cell_1/MatMul_1/ReadVariableOpЂmodel/lstm_1/whileЂ/model/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOpЂ.model/lstm_2/lstm_cell_2/MatMul/ReadVariableOpЂ0model/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOpЂmodel/lstm_2/whileG
model/lstm/ShapeShapeinput_1*
T0*
_output_shapes
:h
model/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 model/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 model/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model/lstm/strided_sliceStridedSlicemodel/lstm/Shape:output:0'model/lstm/strided_slice/stack:output:0)model/lstm/strided_slice/stack_1:output:0)model/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
model/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
model/lstm/zeros/packedPack!model/lstm/strided_slice:output:0"model/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:[
model/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
model/lstm/zerosFill model/lstm/zeros/packed:output:0model/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ]
model/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
model/lstm/zeros_1/packedPack!model/lstm/strided_slice:output:0$model/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:]
model/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
model/lstm/zeros_1Fill"model/lstm/zeros_1/packed:output:0!model/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџn
model/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
model/lstm/transpose	Transposeinput_1"model/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџEZ
model/lstm/Shape_1Shapemodel/lstm/transpose:y:0*
T0*
_output_shapes
:j
 model/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"model/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"model/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model/lstm/strided_slice_1StridedSlicemodel/lstm/Shape_1:output:0)model/lstm/strided_slice_1/stack:output:0+model/lstm/strided_slice_1/stack_1:output:0+model/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
&model/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџе
model/lstm/TensorArrayV2TensorListReserve/model/lstm/TensorArrayV2/element_shape:output:0#model/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
@model/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   
2model/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/lstm/transpose:y:0Imodel/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвj
 model/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"model/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"model/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB: 
model/lstm/strided_slice_2StridedSlicemodel/lstm/transpose:y:0)model/lstm/strided_slice_2/stack:output:0+model/lstm/strided_slice_2/stack_1:output:0+model/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџE*
shrink_axis_mask
*model/lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp3model_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes

:Ex*
dtype0А
model/lstm/lstm_cell/MatMulMatMul#model/lstm/strided_slice_2:output:02model/lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxЂ
,model/lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp5model_lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0Њ
model/lstm/lstm_cell/MatMul_1MatMulmodel/lstm/zeros:output:04model/lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxЃ
model/lstm/lstm_cell/addAddV2%model/lstm/lstm_cell/MatMul:product:0'model/lstm/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџx
+model/lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp4model_lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0Ќ
model/lstm/lstm_cell/BiasAddBiasAddmodel/lstm/lstm_cell/add:z:03model/lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxf
$model/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ѕ
model/lstm/lstm_cell/splitSplit-model/lstm/lstm_cell/split/split_dim:output:0%model/lstm/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split~
model/lstm/lstm_cell/SigmoidSigmoid#model/lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
model/lstm/lstm_cell/Sigmoid_1Sigmoid#model/lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
model/lstm/lstm_cell/mulMul"model/lstm/lstm_cell/Sigmoid_1:y:0model/lstm/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџx
model/lstm/lstm_cell/ReluRelu#model/lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
model/lstm/lstm_cell/mul_1Mul model/lstm/lstm_cell/Sigmoid:y:0'model/lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
model/lstm/lstm_cell/add_1AddV2model/lstm/lstm_cell/mul:z:0model/lstm/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
model/lstm/lstm_cell/Sigmoid_2Sigmoid#model/lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџu
model/lstm/lstm_cell/Relu_1Relumodel/lstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
model/lstm/lstm_cell/mul_2Mul"model/lstm/lstm_cell/Sigmoid_2:y:0)model/lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџy
(model/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   й
model/lstm/TensorArrayV2_1TensorListReserve1model/lstm/TensorArrayV2_1/element_shape:output:0#model/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвQ
model/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : n
#model/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ_
model/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
model/lstm/whileWhile&model/lstm/while/loop_counter:output:0,model/lstm/while/maximum_iterations:output:0model/lstm/time:output:0#model/lstm/TensorArrayV2_1:handle:0model/lstm/zeros:output:0model/lstm/zeros_1:output:0#model/lstm/strided_slice_1:output:0Bmodel/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:03model_lstm_lstm_cell_matmul_readvariableop_resource5model_lstm_lstm_cell_matmul_1_readvariableop_resource4model_lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
model_lstm_while_body_300606*(
cond R
model_lstm_while_cond_300605*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
;model/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   у
-model/lstm/TensorArrayV2Stack/TensorListStackTensorListStackmodel/lstm/while:output:3Dmodel/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0s
 model/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџl
"model/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: l
"model/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
model/lstm/strided_slice_3StridedSlice6model/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)model/lstm/strided_slice_3/stack:output:0+model/lstm/strided_slice_3/stack_1:output:0+model/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskp
model/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          З
model/lstm/transpose_1	Transpose6model/lstm/TensorArrayV2Stack/TensorListStack:tensor:0$model/lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџf
model/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    t
model/dropout/IdentityIdentitymodel/lstm/transpose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџa
model/lstm_1/ShapeShapemodel/dropout/Identity:output:0*
T0*
_output_shapes
:j
 model/lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"model/lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"model/lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model/lstm_1/strided_sliceStridedSlicemodel/lstm_1/Shape:output:0)model/lstm_1/strided_slice/stack:output:0+model/lstm_1/strided_slice/stack_1:output:0+model/lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
model/lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
model/lstm_1/zeros/packedPack#model/lstm_1/strided_slice:output:0$model/lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
model/lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
model/lstm_1/zerosFill"model/lstm_1/zeros/packed:output:0!model/lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ_
model/lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
model/lstm_1/zeros_1/packedPack#model/lstm_1/strided_slice:output:0&model/lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:_
model/lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
model/lstm_1/zeros_1Fill$model/lstm_1/zeros_1/packed:output:0#model/lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџp
model/lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"           
model/lstm_1/transpose	Transposemodel/dropout/Identity:output:0$model/lstm_1/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ^
model/lstm_1/Shape_1Shapemodel/lstm_1/transpose:y:0*
T0*
_output_shapes
:l
"model/lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$model/lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$model/lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model/lstm_1/strided_slice_1StridedSlicemodel/lstm_1/Shape_1:output:0+model/lstm_1/strided_slice_1/stack:output:0-model/lstm_1/strided_slice_1/stack_1:output:0-model/lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(model/lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџл
model/lstm_1/TensorArrayV2TensorListReserve1model/lstm_1/TensorArrayV2/element_shape:output:0%model/lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Bmodel/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
4model/lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/lstm_1/transpose:y:0Kmodel/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвl
"model/lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$model/lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$model/lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
model/lstm_1/strided_slice_2StridedSlicemodel/lstm_1/transpose:y:0+model/lstm_1/strided_slice_2/stack:output:0-model/lstm_1/strided_slice_2/stack_1:output:0-model/lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskІ
.model/lstm_1/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp7model_lstm_1_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0К
model/lstm_1/lstm_cell_1/MatMulMatMul%model/lstm_1/strided_slice_2:output:06model/lstm_1/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPЊ
0model/lstm_1/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp9model_lstm_1_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype0Д
!model/lstm_1/lstm_cell_1/MatMul_1MatMulmodel/lstm_1/zeros:output:08model/lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPЏ
model/lstm_1/lstm_cell_1/addAddV2)model/lstm_1/lstm_cell_1/MatMul:product:0+model/lstm_1/lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџPЄ
/model/lstm_1/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp8model_lstm_1_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0И
 model/lstm_1/lstm_cell_1/BiasAddBiasAdd model/lstm_1/lstm_cell_1/add:z:07model/lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPj
(model/lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
model/lstm_1/lstm_cell_1/splitSplit1model/lstm_1/lstm_cell_1/split/split_dim:output:0)model/lstm_1/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
 model/lstm_1/lstm_cell_1/SigmoidSigmoid'model/lstm_1/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
"model/lstm_1/lstm_cell_1/Sigmoid_1Sigmoid'model/lstm_1/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
model/lstm_1/lstm_cell_1/mulMul&model/lstm_1/lstm_cell_1/Sigmoid_1:y:0model/lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
model/lstm_1/lstm_cell_1/ReluRelu'model/lstm_1/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџЊ
model/lstm_1/lstm_cell_1/mul_1Mul$model/lstm_1/lstm_cell_1/Sigmoid:y:0+model/lstm_1/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
model/lstm_1/lstm_cell_1/add_1AddV2 model/lstm_1/lstm_cell_1/mul:z:0"model/lstm_1/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
"model/lstm_1/lstm_cell_1/Sigmoid_2Sigmoid'model/lstm_1/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ}
model/lstm_1/lstm_cell_1/Relu_1Relu"model/lstm_1/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџЎ
model/lstm_1/lstm_cell_1/mul_2Mul&model/lstm_1/lstm_cell_1/Sigmoid_2:y:0-model/lstm_1/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ{
*model/lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   п
model/lstm_1/TensorArrayV2_1TensorListReserve3model/lstm_1/TensorArrayV2_1/element_shape:output:0%model/lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвS
model/lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%model/lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџa
model/lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Е
model/lstm_1/whileWhile(model/lstm_1/while/loop_counter:output:0.model/lstm_1/while/maximum_iterations:output:0model/lstm_1/time:output:0%model/lstm_1/TensorArrayV2_1:handle:0model/lstm_1/zeros:output:0model/lstm_1/zeros_1:output:0%model/lstm_1/strided_slice_1:output:0Dmodel/lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:07model_lstm_1_lstm_cell_1_matmul_readvariableop_resource9model_lstm_1_lstm_cell_1_matmul_1_readvariableop_resource8model_lstm_1_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( **
body"R 
model_lstm_1_while_body_300746**
cond"R 
model_lstm_1_while_cond_300745*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
=model/lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   щ
/model/lstm_1/TensorArrayV2Stack/TensorListStackTensorListStackmodel/lstm_1/while:output:3Fmodel/lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0u
"model/lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџn
$model/lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$model/lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ш
model/lstm_1/strided_slice_3StridedSlice8model/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0+model/lstm_1/strided_slice_3/stack:output:0-model/lstm_1/strided_slice_3/stack_1:output:0-model/lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskr
model/lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
model/lstm_1/transpose_1	Transpose8model/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0&model/lstm_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџh
model/lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ^
model/lstm_2/ShapeShapemodel/lstm_1/transpose_1:y:0*
T0*
_output_shapes
:j
 model/lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"model/lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"model/lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model/lstm_2/strided_sliceStridedSlicemodel/lstm_2/Shape:output:0)model/lstm_2/strided_slice/stack:output:0+model/lstm_2/strided_slice/stack_1:output:0+model/lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
model/lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

model/lstm_2/zeros/packedPack#model/lstm_2/strided_slice:output:0$model/lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
model/lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
model/lstm_2/zerosFill"model/lstm_2/zeros/packed:output:0!model/lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
_
model/lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

model/lstm_2/zeros_1/packedPack#model/lstm_2/strided_slice:output:0&model/lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:_
model/lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
model/lstm_2/zeros_1Fill$model/lstm_2/zeros_1/packed:output:0#model/lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
p
model/lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
model/lstm_2/transpose	Transposemodel/lstm_1/transpose_1:y:0$model/lstm_2/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ^
model/lstm_2/Shape_1Shapemodel/lstm_2/transpose:y:0*
T0*
_output_shapes
:l
"model/lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$model/lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$model/lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model/lstm_2/strided_slice_1StridedSlicemodel/lstm_2/Shape_1:output:0+model/lstm_2/strided_slice_1/stack:output:0-model/lstm_2/strided_slice_1/stack_1:output:0-model/lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(model/lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџл
model/lstm_2/TensorArrayV2TensorListReserve1model/lstm_2/TensorArrayV2/element_shape:output:0%model/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Bmodel/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
4model/lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/lstm_2/transpose:y:0Kmodel/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвl
"model/lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$model/lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$model/lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
model/lstm_2/strided_slice_2StridedSlicemodel/lstm_2/transpose:y:0+model/lstm_2/strided_slice_2/stack:output:0-model/lstm_2/strided_slice_2/stack_1:output:0-model/lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskІ
.model/lstm_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp7model_lstm_2_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0К
model/lstm_2/lstm_cell_2/MatMulMatMul%model/lstm_2/strided_slice_2:output:06model/lstm_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(Њ
0model/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp9model_lstm_2_lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype0Д
!model/lstm_2/lstm_cell_2/MatMul_1MatMulmodel/lstm_2/zeros:output:08model/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(Џ
model/lstm_2/lstm_cell_2/addAddV2)model/lstm_2/lstm_cell_2/MatMul:product:0+model/lstm_2/lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(Є
/model/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp8model_lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0И
 model/lstm_2/lstm_cell_2/BiasAddBiasAdd model/lstm_2/lstm_cell_2/add:z:07model/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(j
(model/lstm_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
model/lstm_2/lstm_cell_2/splitSplit1model/lstm_2/lstm_cell_2/split/split_dim:output:0)model/lstm_2/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_split
 model/lstm_2/lstm_cell_2/SigmoidSigmoid'model/lstm_2/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ

"model/lstm_2/lstm_cell_2/Sigmoid_1Sigmoid'model/lstm_2/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ

model/lstm_2/lstm_cell_2/mulMul&model/lstm_2/lstm_cell_2/Sigmoid_1:y:0model/lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ

model/lstm_2/lstm_cell_2/ReluRelu'model/lstm_2/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
model/lstm_2/lstm_cell_2/mul_1Mul$model/lstm_2/lstm_cell_2/Sigmoid:y:0+model/lstm_2/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ

model/lstm_2/lstm_cell_2/add_1AddV2 model/lstm_2/lstm_cell_2/mul:z:0"model/lstm_2/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

"model/lstm_2/lstm_cell_2/Sigmoid_2Sigmoid'model/lstm_2/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
}
model/lstm_2/lstm_cell_2/Relu_1Relu"model/lstm_2/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ў
model/lstm_2/lstm_cell_2/mul_2Mul&model/lstm_2/lstm_cell_2/Sigmoid_2:y:0-model/lstm_2/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
{
*model/lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   k
)model/lstm_2/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ь
model/lstm_2/TensorArrayV2_1TensorListReserve3model/lstm_2/TensorArrayV2_1/element_shape:output:02model/lstm_2/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвS
model/lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%model/lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџa
model/lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Е
model/lstm_2/whileWhile(model/lstm_2/while/loop_counter:output:0.model/lstm_2/while/maximum_iterations:output:0model/lstm_2/time:output:0%model/lstm_2/TensorArrayV2_1:handle:0model/lstm_2/zeros:output:0model/lstm_2/zeros_1:output:0%model/lstm_2/strided_slice_1:output:0Dmodel/lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:07model_lstm_2_lstm_cell_2_matmul_readvariableop_resource9model_lstm_2_lstm_cell_2_matmul_1_readvariableop_resource8model_lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( **
body"R 
model_lstm_2_while_body_300886**
cond"R 
model_lstm_2_while_cond_300885*K
output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *
parallel_iterations 
=model/lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   §
/model/lstm_2/TensorArrayV2Stack/TensorListStackTensorListStackmodel/lstm_2/while:output:3Fmodel/lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsu
"model/lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџn
$model/lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$model/lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ш
model/lstm_2/strided_slice_3StridedSlice8model/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0+model/lstm_2/strided_slice_3/stack:output:0-model/lstm_2/strided_slice_3/stack_1:output:0-model/lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maskr
model/lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
model/lstm_2/transpose_1	Transpose8model/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0&model/lstm_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
h
model/lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0 
model/dense/MatMulMatMul%model/lstm_2/strided_slice_3:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ

'model/fingerprint/MatMul/ReadVariableOpReadVariableOp0model_fingerprint_matmul_readvariableop_resource*
_output_shapes

:
@*
dtype0Ѕ
model/fingerprint/MatMulMatMulmodel/dense/Relu:activations:0/model/fingerprint/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
(model/fingerprint/BiasAdd/ReadVariableOpReadVariableOp1model_fingerprint_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ќ
model/fingerprint/BiasAddBiasAdd"model/fingerprint/MatMul:product:00model/fingerprint/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@t
model/fingerprint/ReluRelu"model/fingerprint/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0Ѓ
model/dense_3/MatMulMatMul$model/fingerprint/Relu:activations:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0 
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ l
model/dense_3/ReluRelumodel/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0Ѓ
model/dense_1/MatMulMatMul$model/fingerprint/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0 
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ l
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
model/dense_4/MatMulMatMul model/dense_3/Relu:activations:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџl
model/dense_4/ReluRelumodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
model/dense_2/MatMulMatMul model/dense_1/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџl
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
+model/identity_output/MatMul/ReadVariableOpReadVariableOp4model_identity_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Џ
model/identity_output/MatMulMatMul model/dense_4/Relu:activations:03model/identity_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
,model/identity_output/BiasAdd/ReadVariableOpReadVariableOp5model_identity_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
model/identity_output/BiasAddBiasAdd&model/identity_output/MatMul:product:04model/identity_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
model/identity_output/SoftmaxSoftmax&model/identity_output/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
(model/class_output/MatMul/ReadVariableOpReadVariableOp1model_class_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Љ
model/class_output/MatMulMatMul model/dense_2/Relu:activations:00model/class_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
)model/class_output/BiasAdd/ReadVariableOpReadVariableOp2model_class_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
model/class_output/BiasAddBiasAdd#model/class_output/MatMul:product:01model/class_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|
model/class_output/SoftmaxSoftmax#model/class_output/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
IdentityIdentity$model/class_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџx

Identity_1Identity'model/identity_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџП	
NoOpNoOp*^model/class_output/BiasAdd/ReadVariableOp)^model/class_output/MatMul/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp)^model/fingerprint/BiasAdd/ReadVariableOp(^model/fingerprint/MatMul/ReadVariableOp-^model/identity_output/BiasAdd/ReadVariableOp,^model/identity_output/MatMul/ReadVariableOp,^model/lstm/lstm_cell/BiasAdd/ReadVariableOp+^model/lstm/lstm_cell/MatMul/ReadVariableOp-^model/lstm/lstm_cell/MatMul_1/ReadVariableOp^model/lstm/while0^model/lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp/^model/lstm_1/lstm_cell_1/MatMul/ReadVariableOp1^model/lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp^model/lstm_1/while0^model/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp/^model/lstm_2/lstm_cell_2/MatMul/ReadVariableOp1^model/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp^model/lstm_2/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџE: : : : : : : : : : : : : : : : : : : : : : : : : 2V
)model/class_output/BiasAdd/ReadVariableOp)model/class_output/BiasAdd/ReadVariableOp2T
(model/class_output/MatMul/ReadVariableOp(model/class_output/MatMul/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp2T
(model/fingerprint/BiasAdd/ReadVariableOp(model/fingerprint/BiasAdd/ReadVariableOp2R
'model/fingerprint/MatMul/ReadVariableOp'model/fingerprint/MatMul/ReadVariableOp2\
,model/identity_output/BiasAdd/ReadVariableOp,model/identity_output/BiasAdd/ReadVariableOp2Z
+model/identity_output/MatMul/ReadVariableOp+model/identity_output/MatMul/ReadVariableOp2Z
+model/lstm/lstm_cell/BiasAdd/ReadVariableOp+model/lstm/lstm_cell/BiasAdd/ReadVariableOp2X
*model/lstm/lstm_cell/MatMul/ReadVariableOp*model/lstm/lstm_cell/MatMul/ReadVariableOp2\
,model/lstm/lstm_cell/MatMul_1/ReadVariableOp,model/lstm/lstm_cell/MatMul_1/ReadVariableOp2$
model/lstm/whilemodel/lstm/while2b
/model/lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp/model/lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp2`
.model/lstm_1/lstm_cell_1/MatMul/ReadVariableOp.model/lstm_1/lstm_cell_1/MatMul/ReadVariableOp2d
0model/lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp0model/lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp2(
model/lstm_1/whilemodel/lstm_1/while2b
/model/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp/model/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp2`
.model/lstm_2/lstm_cell_2/MatMul/ReadVariableOp.model/lstm_2/lstm_cell_2/MatMul/ReadVariableOp2d
0model/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp0model/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp2(
model/lstm_2/whilemodel/lstm_2/while:T P
+
_output_shapes
:џџџџџџџџџE
!
_user_specified_name	input_1
ф=
Ѕ

A__inference_model_layer_call_and_return_conditional_losses_302686

inputs
lstm_302232:Ex
lstm_302234:x
lstm_302236:x
lstm_1_302389:P
lstm_1_302391:P
lstm_1_302393:P
lstm_2_302541:(
lstm_2_302543:
(
lstm_2_302545:(
dense_302560:


dense_302562:
$
fingerprint_302577:
@ 
fingerprint_302579:@ 
dense_3_302594:@ 
dense_3_302596:  
dense_1_302611:@ 
dense_1_302613:  
dense_4_302628: 
dense_4_302630: 
dense_2_302645: 
dense_2_302647:(
identity_output_302662:$
identity_output_302664:%
class_output_302679:!
class_output_302681:
identity

identity_1Ђ$class_output/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂ#fingerprint/StatefulPartitionedCallЂ'identity_output/StatefulPartitionedCallЂlstm/StatefulPartitionedCallЂlstm_1/StatefulPartitionedCallЂlstm_2/StatefulPartitionedCallѓ
lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm_302232lstm_302234lstm_302236*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_302231й
dropout/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_302244
lstm_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0lstm_1_302389lstm_1_302391lstm_1_302393*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_302388
lstm_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0lstm_2_302541lstm_2_302543lstm_2_302545*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_302540
dense/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_302560dense_302562*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_302559
#fingerprint/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0fingerprint_302577fingerprint_302579*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_fingerprint_layer_call_and_return_conditional_losses_302576
dense_3/StatefulPartitionedCallStatefulPartitionedCall,fingerprint/StatefulPartitionedCall:output:0dense_3_302594dense_3_302596*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_302593
dense_1/StatefulPartitionedCallStatefulPartitionedCall,fingerprint/StatefulPartitionedCall:output:0dense_1_302611dense_1_302613*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_302610
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_302628dense_4_302630*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_302627
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_302645dense_2_302647*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_302644Ў
'identity_output/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0identity_output_302662identity_output_302664*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_identity_output_layer_call_and_return_conditional_losses_302661Ђ
$class_output/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0class_output_302679class_output_302681*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_class_output_layer_call_and_return_conditional_losses_302678|
IdentityIdentity-class_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_1Identity0identity_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЦ
NoOpNoOp%^class_output/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall$^fingerprint/StatefulPartitionedCall(^identity_output/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџE: : : : : : : : : : : : : : : : : : : : : : : : : 2L
$class_output/StatefulPartitionedCall$class_output/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2J
#fingerprint/StatefulPartitionedCall#fingerprint/StatefulPartitionedCall2R
'identity_output/StatefulPartitionedCall'identity_output/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs
K

B__inference_lstm_2_layer_call_and_return_conditional_losses_306454
inputs_0<
*lstm_cell_2_matmul_readvariableop_resource:(>
,lstm_cell_2_matmul_1_readvariableop_resource:
(9
+lstm_cell_2_biasadd_readvariableop_resource:(
identityЂ"lstm_cell_2/BiasAdd/ReadVariableOpЂ!lstm_cell_2/MatMul/ReadVariableOpЂ#lstm_cell_2/MatMul_1/ReadVariableOpЂwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype0
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :к
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitl
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
u
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
f
lstm_cell_2/ReluRelulstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
x
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
c
lstm_cell_2/Relu_1Relulstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0 lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_306369*
condR
while_cond_306368*K
output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Н
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0


є
C__inference_dense_1_layer_call_and_return_conditional_losses_306804

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ц
a
C__inference_dropout_layer_call_and_return_conditional_losses_305492

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:џџџџџџџџџ_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


b
C__inference_dropout_layer_call_and_return_conditional_losses_303173

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Њ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџs
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Е
У
while_cond_301108
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_301108___redundant_placeholder04
0while_while_cond_301108___redundant_placeholder14
0while_while_cond_301108___redundant_placeholder24
0while_while_cond_301108___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
H
с
model_lstm_2_while_body_3008866
2model_lstm_2_while_model_lstm_2_while_loop_counter<
8model_lstm_2_while_model_lstm_2_while_maximum_iterations"
model_lstm_2_while_placeholder$
 model_lstm_2_while_placeholder_1$
 model_lstm_2_while_placeholder_2$
 model_lstm_2_while_placeholder_35
1model_lstm_2_while_model_lstm_2_strided_slice_1_0q
mmodel_lstm_2_while_tensorarrayv2read_tensorlistgetitem_model_lstm_2_tensorarrayunstack_tensorlistfromtensor_0Q
?model_lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0:(S
Amodel_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0:
(N
@model_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0:(
model_lstm_2_while_identity!
model_lstm_2_while_identity_1!
model_lstm_2_while_identity_2!
model_lstm_2_while_identity_3!
model_lstm_2_while_identity_4!
model_lstm_2_while_identity_53
/model_lstm_2_while_model_lstm_2_strided_slice_1o
kmodel_lstm_2_while_tensorarrayv2read_tensorlistgetitem_model_lstm_2_tensorarrayunstack_tensorlistfromtensorO
=model_lstm_2_while_lstm_cell_2_matmul_readvariableop_resource:(Q
?model_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource:
(L
>model_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource:(Ђ5model/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOpЂ4model/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOpЂ6model/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp
Dmodel/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ч
6model/lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmmodel_lstm_2_while_tensorarrayv2read_tensorlistgetitem_model_lstm_2_tensorarrayunstack_tensorlistfromtensor_0model_lstm_2_while_placeholderMmodel/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Д
4model/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp?model_lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:(*
dtype0о
%model/lstm_2/while/lstm_cell_2/MatMulMatMul=model/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0<model/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(И
6model/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpAmodel_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype0Х
'model/lstm_2/while/lstm_cell_2/MatMul_1MatMul model_lstm_2_while_placeholder_2>model/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(С
"model/lstm_2/while/lstm_cell_2/addAddV2/model/lstm_2/while/lstm_cell_2/MatMul:product:01model/lstm_2/while/lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(В
5model/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp@model_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype0Ъ
&model/lstm_2/while/lstm_cell_2/BiasAddBiasAdd&model/lstm_2/while/lstm_cell_2/add:z:0=model/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(p
.model/lstm_2/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$model/lstm_2/while/lstm_cell_2/splitSplit7model/lstm_2/while/lstm_cell_2/split/split_dim:output:0/model/lstm_2/while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_split
&model/lstm_2/while/lstm_cell_2/SigmoidSigmoid-model/lstm_2/while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ

(model/lstm_2/while/lstm_cell_2/Sigmoid_1Sigmoid-model/lstm_2/while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
Ћ
"model/lstm_2/while/lstm_cell_2/mulMul,model/lstm_2/while/lstm_cell_2/Sigmoid_1:y:0 model_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ

#model/lstm_2/while/lstm_cell_2/ReluRelu-model/lstm_2/while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
М
$model/lstm_2/while/lstm_cell_2/mul_1Mul*model/lstm_2/while/lstm_cell_2/Sigmoid:y:01model/lstm_2/while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
Б
$model/lstm_2/while/lstm_cell_2/add_1AddV2&model/lstm_2/while/lstm_cell_2/mul:z:0(model/lstm_2/while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

(model/lstm_2/while/lstm_cell_2/Sigmoid_2Sigmoid-model/lstm_2/while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ

%model/lstm_2/while/lstm_cell_2/Relu_1Relu(model/lstm_2/while/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
Р
$model/lstm_2/while/lstm_cell_2/mul_2Mul,model/lstm_2/while/lstm_cell_2/Sigmoid_2:y:03model/lstm_2/while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ

=model/lstm_2/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B :  
7model/lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem model_lstm_2_while_placeholder_1Fmodel/lstm_2/while/TensorArrayV2Write/TensorListSetItem/index:output:0(model/lstm_2/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвZ
model/lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
model/lstm_2/while/addAddV2model_lstm_2_while_placeholder!model/lstm_2/while/add/y:output:0*
T0*
_output_shapes
: \
model/lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
model/lstm_2/while/add_1AddV22model_lstm_2_while_model_lstm_2_while_loop_counter#model/lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 
model/lstm_2/while/IdentityIdentitymodel/lstm_2/while/add_1:z:0^model/lstm_2/while/NoOp*
T0*
_output_shapes
: 
model/lstm_2/while/Identity_1Identity8model_lstm_2_while_model_lstm_2_while_maximum_iterations^model/lstm_2/while/NoOp*
T0*
_output_shapes
: 
model/lstm_2/while/Identity_2Identitymodel/lstm_2/while/add:z:0^model/lstm_2/while/NoOp*
T0*
_output_shapes
: ­
model/lstm_2/while/Identity_3IdentityGmodel/lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model/lstm_2/while/NoOp*
T0*
_output_shapes
: 
model/lstm_2/while/Identity_4Identity(model/lstm_2/while/lstm_cell_2/mul_2:z:0^model/lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

model/lstm_2/while/Identity_5Identity(model/lstm_2/while/lstm_cell_2/add_1:z:0^model/lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

model/lstm_2/while/NoOpNoOp6^model/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp5^model/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp7^model/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
model_lstm_2_while_identity$model/lstm_2/while/Identity:output:0"G
model_lstm_2_while_identity_1&model/lstm_2/while/Identity_1:output:0"G
model_lstm_2_while_identity_2&model/lstm_2/while/Identity_2:output:0"G
model_lstm_2_while_identity_3&model/lstm_2/while/Identity_3:output:0"G
model_lstm_2_while_identity_4&model/lstm_2/while/Identity_4:output:0"G
model_lstm_2_while_identity_5&model/lstm_2/while/Identity_5:output:0"
>model_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource@model_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0"
?model_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resourceAmodel_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0"
=model_lstm_2_while_lstm_cell_2_matmul_readvariableop_resource?model_lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0"d
/model_lstm_2_while_model_lstm_2_strided_slice_11model_lstm_2_while_model_lstm_2_strided_slice_1_0"м
kmodel_lstm_2_while_tensorarrayv2read_tensorlistgetitem_model_lstm_2_tensorarrayunstack_tensorlistfromtensormmodel_lstm_2_while_tensorarrayv2read_tensorlistgetitem_model_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : 2n
5model/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp5model/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp2l
4model/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp4model/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp2p
6model/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp6model/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
шF
с
model_lstm_1_while_body_3007466
2model_lstm_1_while_model_lstm_1_while_loop_counter<
8model_lstm_1_while_model_lstm_1_while_maximum_iterations"
model_lstm_1_while_placeholder$
 model_lstm_1_while_placeholder_1$
 model_lstm_1_while_placeholder_2$
 model_lstm_1_while_placeholder_35
1model_lstm_1_while_model_lstm_1_strided_slice_1_0q
mmodel_lstm_1_while_tensorarrayv2read_tensorlistgetitem_model_lstm_1_tensorarrayunstack_tensorlistfromtensor_0Q
?model_lstm_1_while_lstm_cell_1_matmul_readvariableop_resource_0:PS
Amodel_lstm_1_while_lstm_cell_1_matmul_1_readvariableop_resource_0:PN
@model_lstm_1_while_lstm_cell_1_biasadd_readvariableop_resource_0:P
model_lstm_1_while_identity!
model_lstm_1_while_identity_1!
model_lstm_1_while_identity_2!
model_lstm_1_while_identity_3!
model_lstm_1_while_identity_4!
model_lstm_1_while_identity_53
/model_lstm_1_while_model_lstm_1_strided_slice_1o
kmodel_lstm_1_while_tensorarrayv2read_tensorlistgetitem_model_lstm_1_tensorarrayunstack_tensorlistfromtensorO
=model_lstm_1_while_lstm_cell_1_matmul_readvariableop_resource:PQ
?model_lstm_1_while_lstm_cell_1_matmul_1_readvariableop_resource:PL
>model_lstm_1_while_lstm_cell_1_biasadd_readvariableop_resource:PЂ5model/lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOpЂ4model/lstm_1/while/lstm_cell_1/MatMul/ReadVariableOpЂ6model/lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOp
Dmodel/lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ч
6model/lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmmodel_lstm_1_while_tensorarrayv2read_tensorlistgetitem_model_lstm_1_tensorarrayunstack_tensorlistfromtensor_0model_lstm_1_while_placeholderMmodel/lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Д
4model/lstm_1/while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp?model_lstm_1_while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0о
%model/lstm_1/while/lstm_cell_1/MatMulMatMul=model/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0<model/lstm_1/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPИ
6model/lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpAmodel_lstm_1_while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype0Х
'model/lstm_1/while/lstm_cell_1/MatMul_1MatMul model_lstm_1_while_placeholder_2>model/lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPС
"model/lstm_1/while/lstm_cell_1/addAddV2/model/lstm_1/while/lstm_cell_1/MatMul:product:01model/lstm_1/while/lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџPВ
5model/lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp@model_lstm_1_while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0Ъ
&model/lstm_1/while/lstm_cell_1/BiasAddBiasAdd&model/lstm_1/while/lstm_cell_1/add:z:0=model/lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPp
.model/lstm_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$model/lstm_1/while/lstm_cell_1/splitSplit7model/lstm_1/while/lstm_cell_1/split/split_dim:output:0/model/lstm_1/while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
&model/lstm_1/while/lstm_cell_1/SigmoidSigmoid-model/lstm_1/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
(model/lstm_1/while/lstm_cell_1/Sigmoid_1Sigmoid-model/lstm_1/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџЋ
"model/lstm_1/while/lstm_cell_1/mulMul,model/lstm_1/while/lstm_cell_1/Sigmoid_1:y:0 model_lstm_1_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ
#model/lstm_1/while/lstm_cell_1/ReluRelu-model/lstm_1/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџМ
$model/lstm_1/while/lstm_cell_1/mul_1Mul*model/lstm_1/while/lstm_cell_1/Sigmoid:y:01model/lstm_1/while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџБ
$model/lstm_1/while/lstm_cell_1/add_1AddV2&model/lstm_1/while/lstm_cell_1/mul:z:0(model/lstm_1/while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
(model/lstm_1/while/lstm_cell_1/Sigmoid_2Sigmoid-model/lstm_1/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
%model/lstm_1/while/lstm_cell_1/Relu_1Relu(model/lstm_1/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџР
$model/lstm_1/while/lstm_cell_1/mul_2Mul,model/lstm_1/while/lstm_cell_1/Sigmoid_2:y:03model/lstm_1/while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџј
7model/lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem model_lstm_1_while_placeholder_1model_lstm_1_while_placeholder(model/lstm_1/while/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвZ
model/lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
model/lstm_1/while/addAddV2model_lstm_1_while_placeholder!model/lstm_1/while/add/y:output:0*
T0*
_output_shapes
: \
model/lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
model/lstm_1/while/add_1AddV22model_lstm_1_while_model_lstm_1_while_loop_counter#model/lstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
model/lstm_1/while/IdentityIdentitymodel/lstm_1/while/add_1:z:0^model/lstm_1/while/NoOp*
T0*
_output_shapes
: 
model/lstm_1/while/Identity_1Identity8model_lstm_1_while_model_lstm_1_while_maximum_iterations^model/lstm_1/while/NoOp*
T0*
_output_shapes
: 
model/lstm_1/while/Identity_2Identitymodel/lstm_1/while/add:z:0^model/lstm_1/while/NoOp*
T0*
_output_shapes
: ­
model/lstm_1/while/Identity_3IdentityGmodel/lstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model/lstm_1/while/NoOp*
T0*
_output_shapes
: 
model/lstm_1/while/Identity_4Identity(model/lstm_1/while/lstm_cell_1/mul_2:z:0^model/lstm_1/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
model/lstm_1/while/Identity_5Identity(model/lstm_1/while/lstm_cell_1/add_1:z:0^model/lstm_1/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
model/lstm_1/while/NoOpNoOp6^model/lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOp5^model/lstm_1/while/lstm_cell_1/MatMul/ReadVariableOp7^model/lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
model_lstm_1_while_identity$model/lstm_1/while/Identity:output:0"G
model_lstm_1_while_identity_1&model/lstm_1/while/Identity_1:output:0"G
model_lstm_1_while_identity_2&model/lstm_1/while/Identity_2:output:0"G
model_lstm_1_while_identity_3&model/lstm_1/while/Identity_3:output:0"G
model_lstm_1_while_identity_4&model/lstm_1/while/Identity_4:output:0"G
model_lstm_1_while_identity_5&model/lstm_1/while/Identity_5:output:0"
>model_lstm_1_while_lstm_cell_1_biasadd_readvariableop_resource@model_lstm_1_while_lstm_cell_1_biasadd_readvariableop_resource_0"
?model_lstm_1_while_lstm_cell_1_matmul_1_readvariableop_resourceAmodel_lstm_1_while_lstm_cell_1_matmul_1_readvariableop_resource_0"
=model_lstm_1_while_lstm_cell_1_matmul_readvariableop_resource?model_lstm_1_while_lstm_cell_1_matmul_readvariableop_resource_0"d
/model_lstm_1_while_model_lstm_1_strided_slice_11model_lstm_1_while_model_lstm_1_strided_slice_1_0"м
kmodel_lstm_1_while_tensorarrayv2read_tensorlistgetitem_model_lstm_1_tensorarrayunstack_tensorlistfromtensormmodel_lstm_1_while_tensorarrayv2read_tensorlistgetitem_model_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2n
5model/lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOp5model/lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOp2l
4model/lstm_1/while/lstm_cell_1/MatMul/ReadVariableOp4model/lstm_1/while/lstm_cell_1/MatMul/ReadVariableOp2p
6model/lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOp6model/lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
уJ

B__inference_lstm_2_layer_call_and_return_conditional_losses_302540

inputs<
*lstm_cell_2_matmul_readvariableop_resource:(>
,lstm_cell_2_matmul_1_readvariableop_resource:
(9
+lstm_cell_2_biasadd_readvariableop_resource:(
identityЂ"lstm_cell_2/BiasAdd/ReadVariableOpЂ!lstm_cell_2/MatMul/ReadVariableOpЂ#lstm_cell_2/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype0
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :к
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitl
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
u
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
f
lstm_cell_2/ReluRelulstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
x
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
c
lstm_cell_2/Relu_1Relulstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0 lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_302455*
condR
while_cond_302454*K
output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Н
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
§
a
(__inference_dropout_layer_call_fn_305487

inputs
identityЂStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_303173s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
фH

@__inference_lstm_layer_call_and_return_conditional_losses_305477

inputs:
(lstm_cell_matmul_readvariableop_resource:Ex<
*lstm_cell_matmul_1_readvariableop_resource:x7
)lstm_cell_biasadd_readvariableop_resource:x
identityЂ lstm_cell/BiasAdd/ReadVariableOpЂlstm_cell/MatMul/ReadVariableOpЂ!lstm_cell/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџED
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџE*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:Ex*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџx
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџq
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџr
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџj
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : љ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_305393*
condR
while_cond_305392*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Т
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџЗ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџE: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs
ц
ђ
,__inference_lstm_cell_1_layer_call_fn_307019

inputs
states_0
states_1
unknown:P
	unknown_0:P
	unknown_1:P
identity

identity_1

identity_2ЂStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_301445o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/1
љ	
Я
lstm_2_while_cond_304230*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3,
(lstm_2_while_less_lstm_2_strided_slice_1B
>lstm_2_while_lstm_2_while_cond_304230___redundant_placeholder0B
>lstm_2_while_lstm_2_while_cond_304230___redundant_placeholder1B
>lstm_2_while_lstm_2_while_cond_304230___redundant_placeholder2B
>lstm_2_while_lstm_2_while_cond_304230___redundant_placeholder3
lstm_2_while_identity
~
lstm_2/while/LessLesslstm_2_while_placeholder(lstm_2_while_less_lstm_2_strided_slice_1*
T0*
_output_shapes
: Y
lstm_2/while/IdentityIdentitylstm_2/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_2_while_identitylstm_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ
:џџџџџџџџџ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
н?
Ё

lstm_1_while_body_304091*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3)
%lstm_1_while_lstm_1_strided_slice_1_0e
alstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0K
9lstm_1_while_lstm_cell_1_matmul_readvariableop_resource_0:PM
;lstm_1_while_lstm_cell_1_matmul_1_readvariableop_resource_0:PH
:lstm_1_while_lstm_cell_1_biasadd_readvariableop_resource_0:P
lstm_1_while_identity
lstm_1_while_identity_1
lstm_1_while_identity_2
lstm_1_while_identity_3
lstm_1_while_identity_4
lstm_1_while_identity_5'
#lstm_1_while_lstm_1_strided_slice_1c
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorI
7lstm_1_while_lstm_cell_1_matmul_readvariableop_resource:PK
9lstm_1_while_lstm_cell_1_matmul_1_readvariableop_resource:PF
8lstm_1_while_lstm_cell_1_biasadd_readvariableop_resource:PЂ/lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOpЂ.lstm_1/while/lstm_cell_1/MatMul/ReadVariableOpЂ0lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOp
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Щ
0lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0lstm_1_while_placeholderGlstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ј
.lstm_1/while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp9lstm_1_while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0Ь
lstm_1/while/lstm_cell_1/MatMulMatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_1/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPЌ
0lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp;lstm_1_while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype0Г
!lstm_1/while/lstm_cell_1/MatMul_1MatMullstm_1_while_placeholder_28lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPЏ
lstm_1/while/lstm_cell_1/addAddV2)lstm_1/while/lstm_cell_1/MatMul:product:0+lstm_1/while/lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџPІ
/lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp:lstm_1_while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0И
 lstm_1/while/lstm_cell_1/BiasAddBiasAdd lstm_1/while/lstm_cell_1/add:z:07lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPj
(lstm_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_1/while/lstm_cell_1/splitSplit1lstm_1/while/lstm_cell_1/split/split_dim:output:0)lstm_1/while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
 lstm_1/while/lstm_cell_1/SigmoidSigmoid'lstm_1/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
"lstm_1/while/lstm_cell_1/Sigmoid_1Sigmoid'lstm_1/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_1/while/lstm_cell_1/mulMul&lstm_1/while/lstm_cell_1/Sigmoid_1:y:0lstm_1_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_1/while/lstm_cell_1/ReluRelu'lstm_1/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџЊ
lstm_1/while/lstm_cell_1/mul_1Mul$lstm_1/while/lstm_cell_1/Sigmoid:y:0+lstm_1/while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_1/while/lstm_cell_1/add_1AddV2 lstm_1/while/lstm_cell_1/mul:z:0"lstm_1/while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
"lstm_1/while/lstm_cell_1/Sigmoid_2Sigmoid'lstm_1/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ}
lstm_1/while/lstm_cell_1/Relu_1Relu"lstm_1/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџЎ
lstm_1/while/lstm_cell_1/mul_2Mul&lstm_1/while/lstm_cell_1/Sigmoid_2:y:0-lstm_1/while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџр
1lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_1_while_placeholder_1lstm_1_while_placeholder"lstm_1/while/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвT
lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_1/while/addAddV2lstm_1_while_placeholderlstm_1/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_1/while/add_1AddV2&lstm_1_while_lstm_1_while_loop_counterlstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_1/while/IdentityIdentitylstm_1/while/add_1:z:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 
lstm_1/while/Identity_1Identity,lstm_1_while_lstm_1_while_maximum_iterations^lstm_1/while/NoOp*
T0*
_output_shapes
: n
lstm_1/while/Identity_2Identitylstm_1/while/add:z:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 
lstm_1/while/Identity_3IdentityAlstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 
lstm_1/while/Identity_4Identity"lstm_1/while/lstm_cell_1/mul_2:z:0^lstm_1/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_1/while/Identity_5Identity"lstm_1/while/lstm_cell_1/add_1:z:0^lstm_1/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџщ
lstm_1/while/NoOpNoOp0^lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOp/^lstm_1/while/lstm_cell_1/MatMul/ReadVariableOp1^lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_1_while_identitylstm_1/while/Identity:output:0";
lstm_1_while_identity_1 lstm_1/while/Identity_1:output:0";
lstm_1_while_identity_2 lstm_1/while/Identity_2:output:0";
lstm_1_while_identity_3 lstm_1/while/Identity_3:output:0";
lstm_1_while_identity_4 lstm_1/while/Identity_4:output:0";
lstm_1_while_identity_5 lstm_1/while/Identity_5:output:0"L
#lstm_1_while_lstm_1_strided_slice_1%lstm_1_while_lstm_1_strided_slice_1_0"v
8lstm_1_while_lstm_cell_1_biasadd_readvariableop_resource:lstm_1_while_lstm_cell_1_biasadd_readvariableop_resource_0"x
9lstm_1_while_lstm_cell_1_matmul_1_readvariableop_resource;lstm_1_while_lstm_cell_1_matmul_1_readvariableop_resource_0"t
7lstm_1_while_lstm_cell_1_matmul_readvariableop_resource9lstm_1_while_lstm_cell_1_matmul_readvariableop_resource_0"Ф
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensoralstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2b
/lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOp/lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOp2`
.lstm_1/while/lstm_cell_1/MatMul/ReadVariableOp.lstm_1/while/lstm_cell_1/MatMul/ReadVariableOp2d
0lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOp0lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Л7
С
while_body_305893
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_1_matmul_readvariableop_resource_0:PF
4while_lstm_cell_1_matmul_1_readvariableop_resource_0:PA
3while_lstm_cell_1_biasadd_readvariableop_resource_0:P
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_1_matmul_readvariableop_resource:PD
2while_lstm_cell_1_matmul_1_readvariableop_resource:P?
1while_lstm_cell_1_biasadd_readvariableop_resource:PЂ(while/lstm_cell_1/BiasAdd/ReadVariableOpЂ'while/lstm_cell_1/MatMul/ReadVariableOpЂ)while/lstm_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
'while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0З
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
)while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype0
while/lstm_cell_1/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
while/lstm_cell_1/addAddV2"while/lstm_cell_1/MatMul:product:0$while/lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP
(while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0Ѓ
while/lstm_cell_1/BiasAddBiasAddwhile/lstm_cell_1/add:z:00while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPc
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ь
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0"while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitx
while/lstm_cell_1/SigmoidSigmoid while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџz
while/lstm_cell_1/Sigmoid_1Sigmoid while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mulMulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџr
while/lstm_cell_1/ReluRelu while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mul_1Mulwhile/lstm_cell_1/Sigmoid:y:0$while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/add_1AddV2while/lstm_cell_1/mul:z:0while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџz
while/lstm_cell_1/Sigmoid_2Sigmoid while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџo
while/lstm_cell_1/Relu_1Reluwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mul_2Mulwhile/lstm_cell_1/Sigmoid_2:y:0&while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_1/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџx
while/Identity_5Identitywhile/lstm_cell_1/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџЭ

while/NoOpNoOp)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_1_biasadd_readvariableop_resource3while_lstm_cell_1_biasadd_readvariableop_resource_0"j
2while_lstm_cell_1_matmul_1_readvariableop_resource4while_lstm_cell_1_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_1_matmul_readvariableop_resource2while_lstm_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2T
(while/lstm_cell_1/BiasAdd/ReadVariableOp(while/lstm_cell_1/BiasAdd/ReadVariableOp2R
'while/lstm_cell_1/MatMul/ReadVariableOp'while/lstm_cell_1/MatMul/ReadVariableOp2V
)while/lstm_cell_1/MatMul_1/ReadVariableOp)while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Є
Г
'__inference_lstm_1_layer_call_fn_305526
inputs_0
unknown:P
	unknown_0:P
	unknown_1:P
identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_301719|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Р

(__inference_dense_4_layer_call_fn_306853

inputs
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_302627o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


є
C__inference_dense_4_layer_call_and_return_conditional_losses_302627

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


є
C__inference_dense_3_layer_call_and_return_conditional_losses_302593

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
т
№
*__inference_lstm_cell_layer_call_fn_306921

inputs
states_0
states_1
unknown:Ex
	unknown_0:x
	unknown_1:x
identity

identity_1

identity_2ЂStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_301095o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџE:џџџџџџџџџ:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/1
Е
У
while_cond_305892
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_305892___redundant_placeholder04
0while_while_cond_305892___redundant_placeholder14
0while_while_cond_305892___redundant_placeholder24
0while_while_cond_305892___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:


є
C__inference_dense_3_layer_call_and_return_conditional_losses_306824

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
в

G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_307166

inputs
states_0
states_10
matmul_readvariableop_resource:(2
 matmul_1_readvariableop_resource:
(-
biasadd_readvariableop_resource:(
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ
N
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
states/1
Р

(__inference_dense_3_layer_call_fn_306813

inputs
unknown:@ 
	unknown_0: 
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_302593o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


ђ
A__inference_dense_layer_call_and_return_conditional_losses_306764

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
з8
С
while_body_302894
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_2_matmul_readvariableop_resource_0:(F
4while_lstm_cell_2_matmul_1_readvariableop_resource_0:
(A
3while_lstm_cell_2_biasadd_readvariableop_resource_0:(
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_2_matmul_readvariableop_resource:(D
2while_lstm_cell_2_matmul_1_readvariableop_resource:
(?
1while_lstm_cell_2_biasadd_readvariableop_resource:(Ђ(while/lstm_cell_2/BiasAdd/ReadVariableOpЂ'while/lstm_cell_2/MatMul/ReadVariableOpЂ)while/lstm_cell_2/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:(*
dtype0З
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype0
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype0Ѓ
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(c
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ь
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitx
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
z
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ
r
while/lstm_cell_2/ReluRelu while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0$while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
z
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
o
while/lstm_cell_2/Relu_1Reluwhile/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0&while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
x
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Э

while/NoOpNoOp)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : 2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
Е
У
while_cond_301299
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_301299___redundant_placeholder04
0while_while_cond_301299___redundant_placeholder14
0while_while_cond_301299___redundant_placeholder24
0while_while_cond_301299___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
т
№
*__inference_lstm_cell_layer_call_fn_306938

inputs
states_0
states_1
unknown:Ex
	unknown_0:x
	unknown_1:x
identity

identity_1

identity_2ЂStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_301241o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџE:џџџџџџџџџ:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/1


є
C__inference_dense_1_layer_call_and_return_conditional_losses_302610

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
в

G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_307100

inputs
states_0
states_10
matmul_readvariableop_resource:P2
 matmul_1_readvariableop_resource:P-
biasadd_readvariableop_resource:P
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџPr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџN
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/1
љ	
Я
lstm_1_while_cond_304090*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3,
(lstm_1_while_less_lstm_1_strided_slice_1B
>lstm_1_while_lstm_1_while_cond_304090___redundant_placeholder0B
>lstm_1_while_lstm_1_while_cond_304090___redundant_placeholder1B
>lstm_1_while_lstm_1_while_cond_304090___redundant_placeholder2B
>lstm_1_while_lstm_1_while_cond_304090___redundant_placeholder3
lstm_1_while_identity
~
lstm_1/while/LessLesslstm_1_while_placeholder(lstm_1_while_less_lstm_1_strided_slice_1*
T0*
_output_shapes
: Y
lstm_1/while/IdentityIdentitylstm_1/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_1_while_identitylstm_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
уJ

B__inference_lstm_2_layer_call_and_return_conditional_losses_306599

inputs<
*lstm_cell_2_matmul_readvariableop_resource:(>
,lstm_cell_2_matmul_1_readvariableop_resource:
(9
+lstm_cell_2_biasadd_readvariableop_resource:(
identityЂ"lstm_cell_2/BiasAdd/ReadVariableOpЂ!lstm_cell_2/MatMul/ReadVariableOpЂ#lstm_cell_2/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype0
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :к
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitl
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
u
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
f
lstm_cell_2/ReluRelulstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
x
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
c
lstm_cell_2/Relu_1Relulstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0 lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_306514*
condR
while_cond_306513*K
output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Н
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ъ

G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_301943

inputs

states
states_10
matmul_readvariableop_resource:(2
 matmul_1_readvariableop_resource:
(-
biasadd_readvariableop_resource:(
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ
N
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_namestates
Е
У
while_cond_304963
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_304963___redundant_placeholder04
0while_while_cond_304963___redundant_placeholder14
0while_while_cond_304963___redundant_placeholder24
0while_while_cond_304963___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Ъ

G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_301591

inputs

states
states_10
matmul_readvariableop_resource:P2
 matmul_1_readvariableop_resource:P-
biasadd_readvariableop_resource:P
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџPr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџN
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates


ј
G__inference_fingerprint_layer_call_and_return_conditional_losses_306784

inputs0
matmul_readvariableop_resource:
@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
њ
Б
'__inference_lstm_1_layer_call_fn_305548

inputs
unknown:P
	unknown_0:P
	unknown_1:P
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_303144s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
фH

@__inference_lstm_layer_call_and_return_conditional_losses_305334

inputs:
(lstm_cell_matmul_readvariableop_resource:Ex<
*lstm_cell_matmul_1_readvariableop_resource:x7
)lstm_cell_biasadd_readvariableop_resource:x
identityЂ lstm_cell/BiasAdd/ReadVariableOpЂlstm_cell/MatMul/ReadVariableOpЂ!lstm_cell/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџED
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџE*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:Ex*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџx
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџq
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџr
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџj
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : љ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_305250*
condR
while_cond_305249*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Т
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџЗ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџE: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs


є
C__inference_dense_4_layer_call_and_return_conditional_losses_306864

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Р

(__inference_dense_1_layer_call_fn_306793

inputs
unknown:@ 
	unknown_0: 
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_302610o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
нI

B__inference_lstm_1_layer_call_and_return_conditional_losses_303144

inputs<
*lstm_cell_1_matmul_readvariableop_resource:P>
,lstm_cell_1_matmul_1_readvariableop_resource:P9
+lstm_cell_1_biasadd_readvariableop_resource:P
identityЂ"lstm_cell_1/BiasAdd/ReadVariableOpЂ!lstm_cell_1/MatMul/ReadVariableOpЂ#lstm_cell_1/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype0
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :к
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitl
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџn
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџu
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
lstm_cell_1/ReluRelulstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџx
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџc
lstm_cell_1/Relu_1Relulstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0 lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_1_matmul_readvariableop_resource,lstm_cell_1_matmul_1_readvariableop_resource+lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_303060*
condR
while_cond_303059*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Т
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџН
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
фH

@__inference_lstm_layer_call_and_return_conditional_losses_303332

inputs:
(lstm_cell_matmul_readvariableop_resource:Ex<
*lstm_cell_matmul_1_readvariableop_resource:x7
)lstm_cell_biasadd_readvariableop_resource:x
identityЂ lstm_cell/BiasAdd/ReadVariableOpЂlstm_cell/MatMul/ReadVariableOpЂ!lstm_cell/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџED
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџE*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:Ex*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџx
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџq
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџr
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџj
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : љ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_303248*
condR
while_cond_303247*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Т
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџЗ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџE: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs
Ъ

G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_301795

inputs

states
states_10
matmul_readvariableop_resource:(2
 matmul_1_readvariableop_resource:
(-
biasadd_readvariableop_resource:(
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ
N
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_namestates
ц
a
C__inference_dropout_layer_call_and_return_conditional_losses_302244

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:џџџџџџџџџ_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
щ
Ѕ
&__inference_model_layer_call_fn_303579
input_1
unknown:Ex
	unknown_0:x
	unknown_1:x
	unknown_2:P
	unknown_3:P
	unknown_4:P
	unknown_5:(
	unknown_6:
(
	unknown_7:(
	unknown_8:


	unknown_9:


unknown_10:
@

unknown_11:@

unknown_12:@ 

unknown_13: 

unknown_14:@ 

unknown_15: 

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:
identity

identity_1ЂStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_303467o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџE: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџE
!
_user_specified_name	input_1
ЎC

model_lstm_while_body_3006062
.model_lstm_while_model_lstm_while_loop_counter8
4model_lstm_while_model_lstm_while_maximum_iterations 
model_lstm_while_placeholder"
model_lstm_while_placeholder_1"
model_lstm_while_placeholder_2"
model_lstm_while_placeholder_31
-model_lstm_while_model_lstm_strided_slice_1_0m
imodel_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensor_0M
;model_lstm_while_lstm_cell_matmul_readvariableop_resource_0:ExO
=model_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:xJ
<model_lstm_while_lstm_cell_biasadd_readvariableop_resource_0:x
model_lstm_while_identity
model_lstm_while_identity_1
model_lstm_while_identity_2
model_lstm_while_identity_3
model_lstm_while_identity_4
model_lstm_while_identity_5/
+model_lstm_while_model_lstm_strided_slice_1k
gmodel_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensorK
9model_lstm_while_lstm_cell_matmul_readvariableop_resource:ExM
;model_lstm_while_lstm_cell_matmul_1_readvariableop_resource:xH
:model_lstm_while_lstm_cell_biasadd_readvariableop_resource:xЂ1model/lstm/while/lstm_cell/BiasAdd/ReadVariableOpЂ0model/lstm/while/lstm_cell/MatMul/ReadVariableOpЂ2model/lstm/while/lstm_cell/MatMul_1/ReadVariableOp
Bmodel/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   н
4model/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemimodel_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensor_0model_lstm_while_placeholderKmodel/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџE*
element_dtype0Ќ
0model/lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp;model_lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:Ex*
dtype0д
!model/lstm/while/lstm_cell/MatMulMatMul;model/lstm/while/TensorArrayV2Read/TensorListGetItem:item:08model/lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxА
2model/lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp=model_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0Л
#model/lstm/while/lstm_cell/MatMul_1MatMulmodel_lstm_while_placeholder_2:model/lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxЕ
model/lstm/while/lstm_cell/addAddV2+model/lstm/while/lstm_cell/MatMul:product:0-model/lstm/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџxЊ
1model/lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp<model_lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0О
"model/lstm/while/lstm_cell/BiasAddBiasAdd"model/lstm/while/lstm_cell/add:z:09model/lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxl
*model/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 model/lstm/while/lstm_cell/splitSplit3model/lstm/while/lstm_cell/split/split_dim:output:0+model/lstm/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
"model/lstm/while/lstm_cell/SigmoidSigmoid)model/lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
$model/lstm/while/lstm_cell/Sigmoid_1Sigmoid)model/lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџЁ
model/lstm/while/lstm_cell/mulMul(model/lstm/while/lstm_cell/Sigmoid_1:y:0model_lstm_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ
model/lstm/while/lstm_cell/ReluRelu)model/lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџА
 model/lstm/while/lstm_cell/mul_1Mul&model/lstm/while/lstm_cell/Sigmoid:y:0-model/lstm/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџЅ
 model/lstm/while/lstm_cell/add_1AddV2"model/lstm/while/lstm_cell/mul:z:0$model/lstm/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
$model/lstm/while/lstm_cell/Sigmoid_2Sigmoid)model/lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
!model/lstm/while/lstm_cell/Relu_1Relu$model/lstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџД
 model/lstm/while/lstm_cell/mul_2Mul(model/lstm/while/lstm_cell/Sigmoid_2:y:0/model/lstm/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџю
5model/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmodel_lstm_while_placeholder_1model_lstm_while_placeholder$model/lstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвX
model/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :}
model/lstm/while/addAddV2model_lstm_while_placeholdermodel/lstm/while/add/y:output:0*
T0*
_output_shapes
: Z
model/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
model/lstm/while/add_1AddV2.model_lstm_while_model_lstm_while_loop_counter!model/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: z
model/lstm/while/IdentityIdentitymodel/lstm/while/add_1:z:0^model/lstm/while/NoOp*
T0*
_output_shapes
: 
model/lstm/while/Identity_1Identity4model_lstm_while_model_lstm_while_maximum_iterations^model/lstm/while/NoOp*
T0*
_output_shapes
: z
model/lstm/while/Identity_2Identitymodel/lstm/while/add:z:0^model/lstm/while/NoOp*
T0*
_output_shapes
: Ї
model/lstm/while/Identity_3IdentityEmodel/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model/lstm/while/NoOp*
T0*
_output_shapes
: 
model/lstm/while/Identity_4Identity$model/lstm/while/lstm_cell/mul_2:z:0^model/lstm/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
model/lstm/while/Identity_5Identity$model/lstm/while/lstm_cell/add_1:z:0^model/lstm/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџѓ
model/lstm/while/NoOpNoOp2^model/lstm/while/lstm_cell/BiasAdd/ReadVariableOp1^model/lstm/while/lstm_cell/MatMul/ReadVariableOp3^model/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
model_lstm_while_identity"model/lstm/while/Identity:output:0"C
model_lstm_while_identity_1$model/lstm/while/Identity_1:output:0"C
model_lstm_while_identity_2$model/lstm/while/Identity_2:output:0"C
model_lstm_while_identity_3$model/lstm/while/Identity_3:output:0"C
model_lstm_while_identity_4$model/lstm/while/Identity_4:output:0"C
model_lstm_while_identity_5$model/lstm/while/Identity_5:output:0"z
:model_lstm_while_lstm_cell_biasadd_readvariableop_resource<model_lstm_while_lstm_cell_biasadd_readvariableop_resource_0"|
;model_lstm_while_lstm_cell_matmul_1_readvariableop_resource=model_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"x
9model_lstm_while_lstm_cell_matmul_readvariableop_resource;model_lstm_while_lstm_cell_matmul_readvariableop_resource_0"\
+model_lstm_while_model_lstm_strided_slice_1-model_lstm_while_model_lstm_strided_slice_1_0"д
gmodel_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensorimodel_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2f
1model/lstm/while/lstm_cell/BiasAdd/ReadVariableOp1model/lstm/while/lstm_cell/BiasAdd/ReadVariableOp2d
0model/lstm/while/lstm_cell/MatMul/ReadVariableOp0model/lstm/while/lstm_cell/MatMul/ReadVariableOp2h
2model/lstm/while/lstm_cell/MatMul_1/ReadVariableOp2model/lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
н?
Ё

lstm_1_while_body_304579*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3)
%lstm_1_while_lstm_1_strided_slice_1_0e
alstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0K
9lstm_1_while_lstm_cell_1_matmul_readvariableop_resource_0:PM
;lstm_1_while_lstm_cell_1_matmul_1_readvariableop_resource_0:PH
:lstm_1_while_lstm_cell_1_biasadd_readvariableop_resource_0:P
lstm_1_while_identity
lstm_1_while_identity_1
lstm_1_while_identity_2
lstm_1_while_identity_3
lstm_1_while_identity_4
lstm_1_while_identity_5'
#lstm_1_while_lstm_1_strided_slice_1c
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorI
7lstm_1_while_lstm_cell_1_matmul_readvariableop_resource:PK
9lstm_1_while_lstm_cell_1_matmul_1_readvariableop_resource:PF
8lstm_1_while_lstm_cell_1_biasadd_readvariableop_resource:PЂ/lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOpЂ.lstm_1/while/lstm_cell_1/MatMul/ReadVariableOpЂ0lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOp
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Щ
0lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0lstm_1_while_placeholderGlstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ј
.lstm_1/while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp9lstm_1_while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0Ь
lstm_1/while/lstm_cell_1/MatMulMatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_1/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPЌ
0lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp;lstm_1_while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype0Г
!lstm_1/while/lstm_cell_1/MatMul_1MatMullstm_1_while_placeholder_28lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPЏ
lstm_1/while/lstm_cell_1/addAddV2)lstm_1/while/lstm_cell_1/MatMul:product:0+lstm_1/while/lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџPІ
/lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp:lstm_1_while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0И
 lstm_1/while/lstm_cell_1/BiasAddBiasAdd lstm_1/while/lstm_cell_1/add:z:07lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPj
(lstm_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_1/while/lstm_cell_1/splitSplit1lstm_1/while/lstm_cell_1/split/split_dim:output:0)lstm_1/while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split
 lstm_1/while/lstm_cell_1/SigmoidSigmoid'lstm_1/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
"lstm_1/while/lstm_cell_1/Sigmoid_1Sigmoid'lstm_1/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_1/while/lstm_cell_1/mulMul&lstm_1/while/lstm_cell_1/Sigmoid_1:y:0lstm_1_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_1/while/lstm_cell_1/ReluRelu'lstm_1/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџЊ
lstm_1/while/lstm_cell_1/mul_1Mul$lstm_1/while/lstm_cell_1/Sigmoid:y:0+lstm_1/while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_1/while/lstm_cell_1/add_1AddV2 lstm_1/while/lstm_cell_1/mul:z:0"lstm_1/while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
"lstm_1/while/lstm_cell_1/Sigmoid_2Sigmoid'lstm_1/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ}
lstm_1/while/lstm_cell_1/Relu_1Relu"lstm_1/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџЎ
lstm_1/while/lstm_cell_1/mul_2Mul&lstm_1/while/lstm_cell_1/Sigmoid_2:y:0-lstm_1/while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџр
1lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_1_while_placeholder_1lstm_1_while_placeholder"lstm_1/while/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвT
lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_1/while/addAddV2lstm_1_while_placeholderlstm_1/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_1/while/add_1AddV2&lstm_1_while_lstm_1_while_loop_counterlstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_1/while/IdentityIdentitylstm_1/while/add_1:z:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 
lstm_1/while/Identity_1Identity,lstm_1_while_lstm_1_while_maximum_iterations^lstm_1/while/NoOp*
T0*
_output_shapes
: n
lstm_1/while/Identity_2Identitylstm_1/while/add:z:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 
lstm_1/while/Identity_3IdentityAlstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 
lstm_1/while/Identity_4Identity"lstm_1/while/lstm_cell_1/mul_2:z:0^lstm_1/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_1/while/Identity_5Identity"lstm_1/while/lstm_cell_1/add_1:z:0^lstm_1/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџщ
lstm_1/while/NoOpNoOp0^lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOp/^lstm_1/while/lstm_cell_1/MatMul/ReadVariableOp1^lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_1_while_identitylstm_1/while/Identity:output:0";
lstm_1_while_identity_1 lstm_1/while/Identity_1:output:0";
lstm_1_while_identity_2 lstm_1/while/Identity_2:output:0";
lstm_1_while_identity_3 lstm_1/while/Identity_3:output:0";
lstm_1_while_identity_4 lstm_1/while/Identity_4:output:0";
lstm_1_while_identity_5 lstm_1/while/Identity_5:output:0"L
#lstm_1_while_lstm_1_strided_slice_1%lstm_1_while_lstm_1_strided_slice_1_0"v
8lstm_1_while_lstm_cell_1_biasadd_readvariableop_resource:lstm_1_while_lstm_cell_1_biasadd_readvariableop_resource_0"x
9lstm_1_while_lstm_cell_1_matmul_1_readvariableop_resource;lstm_1_while_lstm_cell_1_matmul_1_readvariableop_resource_0"t
7lstm_1_while_lstm_cell_1_matmul_readvariableop_resource9lstm_1_while_lstm_cell_1_matmul_readvariableop_resource_0"Ф
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensoralstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2b
/lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOp/lstm_1/while/lstm_cell_1/BiasAdd/ReadVariableOp2`
.lstm_1/while/lstm_cell_1/MatMul/ReadVariableOp.lstm_1/while/lstm_cell_1/MatMul/ReadVariableOp2d
0lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOp0lstm_1/while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
з8
С
while_body_302455
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_2_matmul_readvariableop_resource_0:(F
4while_lstm_cell_2_matmul_1_readvariableop_resource_0:
(A
3while_lstm_cell_2_biasadd_readvariableop_resource_0:(
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_2_matmul_readvariableop_resource:(D
2while_lstm_cell_2_matmul_1_readvariableop_resource:
(?
1while_lstm_cell_2_biasadd_readvariableop_resource:(Ђ(while/lstm_cell_2/BiasAdd/ReadVariableOpЂ'while/lstm_cell_2/MatMul/ReadVariableOpЂ)while/lstm_cell_2/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:(*
dtype0З
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype0
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype0Ѓ
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(c
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ь
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitx
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
z
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ
r
while/lstm_cell_2/ReluRelu while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0$while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
z
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
o
while/lstm_cell_2/Relu_1Reluwhile/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0&while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
x
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Э

while/NoOpNoOp)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : 2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
Е
У
while_cond_305749
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_305749___redundant_placeholder04
0while_while_cond_305749___redundant_placeholder14
0while_while_cond_305749___redundant_placeholder24
0while_while_cond_305749___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Г6
Џ
while_body_304964
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:ExD
2while_lstm_cell_matmul_1_readvariableop_resource_0:x?
1while_lstm_cell_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:ExB
0while_lstm_cell_matmul_1_readvariableop_resource:x=
/while_lstm_cell_biasadd_readvariableop_resource:xЂ&while/lstm_cell/BiasAdd/ReadVariableOpЂ%while/lstm_cell/MatMul/ReadVariableOpЂ'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџE*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:Ex*
dtype0Г
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџx
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџv
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџn
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџv
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџk
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџТ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџv
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџЧ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
фH

@__inference_lstm_layer_call_and_return_conditional_losses_302231

inputs:
(lstm_cell_matmul_readvariableop_resource:Ex<
*lstm_cell_matmul_1_readvariableop_resource:x7
)lstm_cell_biasadd_readvariableop_resource:x
identityЂ lstm_cell/BiasAdd/ReadVariableOpЂlstm_cell/MatMul/ReadVariableOpЂ!lstm_cell/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџED
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџE*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:Ex*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџx
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџq
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџr
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџj
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : љ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_302147*
condR
while_cond_302146*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Т
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџЗ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџE: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs
Ї

ќ
K__inference_identity_output_layer_call_and_return_conditional_losses_302661

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ8
§
B__inference_lstm_2_layer_call_and_return_conditional_losses_302073

inputs$
lstm_cell_2_301989:($
lstm_cell_2_301991:
( 
lstm_cell_2_301993:(
identityЂ#lstm_cell_2/StatefulPartitionedCallЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask№
#lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_2_301989lstm_cell_2_301991lstm_cell_2_301993*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_301943n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Д
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_2_301989lstm_cell_2_301991lstm_cell_2_301993*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_302003*
condR
while_cond_302002*K
output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
t
NoOpNoOp$^lstm_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2J
#lstm_cell_2/StatefulPartitionedCall#lstm_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ц
ђ
,__inference_lstm_cell_2_layer_call_fn_307134

inputs
states_0
states_1
unknown:(
	unknown_0:
(
	unknown_1:(
identity

identity_1

identity_2ЂStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_301943o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
states/1
Е
У
while_cond_302893
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_302893___redundant_placeholder04
0while_while_cond_302893___redundant_placeholder14
0while_while_cond_302893___redundant_placeholder24
0while_while_cond_302893___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ
:џџџџџџџџџ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
Л7
С
while_body_302304
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_1_matmul_readvariableop_resource_0:PF
4while_lstm_cell_1_matmul_1_readvariableop_resource_0:PA
3while_lstm_cell_1_biasadd_readvariableop_resource_0:P
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_1_matmul_readvariableop_resource:PD
2while_lstm_cell_1_matmul_1_readvariableop_resource:P?
1while_lstm_cell_1_biasadd_readvariableop_resource:PЂ(while/lstm_cell_1/BiasAdd/ReadVariableOpЂ'while/lstm_cell_1/MatMul/ReadVariableOpЂ)while/lstm_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
'while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0З
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
)while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype0
while/lstm_cell_1/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
while/lstm_cell_1/addAddV2"while/lstm_cell_1/MatMul:product:0$while/lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP
(while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0Ѓ
while/lstm_cell_1/BiasAddBiasAddwhile/lstm_cell_1/add:z:00while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPc
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ь
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0"while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitx
while/lstm_cell_1/SigmoidSigmoid while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџz
while/lstm_cell_1/Sigmoid_1Sigmoid while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mulMulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџr
while/lstm_cell_1/ReluRelu while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mul_1Mulwhile/lstm_cell_1/Sigmoid:y:0$while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/add_1AddV2while/lstm_cell_1/mul:z:0while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџz
while/lstm_cell_1/Sigmoid_2Sigmoid while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџo
while/lstm_cell_1/Relu_1Reluwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mul_2Mulwhile/lstm_cell_1/Sigmoid_2:y:0&while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_1/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџx
while/Identity_5Identitywhile/lstm_cell_1/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџЭ

while/NoOpNoOp)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_1_biasadd_readvariableop_resource3while_lstm_cell_1_biasadd_readvariableop_resource_0"j
2while_lstm_cell_1_matmul_1_readvariableop_resource4while_lstm_cell_1_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_1_matmul_readvariableop_resource2while_lstm_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2T
(while/lstm_cell_1/BiasAdd/ReadVariableOp(while/lstm_cell_1/BiasAdd/ReadVariableOp2R
'while/lstm_cell_1/MatMul/ReadVariableOp'while/lstm_cell_1/MatMul/ReadVariableOp2V
)while/lstm_cell_1/MatMul_1/ReadVariableOp)while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Ъ

-__inference_class_output_layer_call_fn_306873

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_class_output_layer_call_and_return_conditional_losses_302678o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Е
У
while_cond_305606
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_305606___redundant_placeholder04
0while_while_cond_305606___redundant_placeholder14
0while_while_cond_305606___redundant_placeholder24
0while_while_cond_305606___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
J

B__inference_lstm_1_layer_call_and_return_conditional_losses_305691
inputs_0<
*lstm_cell_1_matmul_readvariableop_resource:P>
,lstm_cell_1_matmul_1_readvariableop_resource:P9
+lstm_cell_1_biasadd_readvariableop_resource:P
identityЂ"lstm_cell_1/BiasAdd/ReadVariableOpЂ!lstm_cell_1/MatMul/ReadVariableOpЂ#lstm_cell_1/MatMul_1/ReadVariableOpЂwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype0
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :к
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitl
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџn
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџu
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
lstm_cell_1/ReluRelulstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџx
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџc
lstm_cell_1/Relu_1Relulstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0 lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_1_matmul_readvariableop_resource,lstm_cell_1_matmul_1_readvariableop_resource+lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_305607*
condR
while_cond_305606*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџН
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Ч
Ѓ
$__inference_signature_wrapper_303778
input_1
unknown:Ex
	unknown_0:x
	unknown_1:x
	unknown_2:P
	unknown_3:P
	unknown_4:P
	unknown_5:(
	unknown_6:
(
	unknown_7:(
	unknown_8:


	unknown_9:


unknown_10:
@

unknown_11:@

unknown_12:@ 

unknown_13: 

unknown_14:@ 

unknown_15: 

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:
identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_301028o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџE: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџE
!
_user_specified_name	input_1


є
C__inference_dense_2_layer_call_and_return_conditional_losses_306844

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ш
ў
E__inference_lstm_cell_layer_call_and_return_conditional_losses_301241

inputs

states
states_10
matmul_readvariableop_resource:Ex2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Ex*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџN
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџE:џџџџџџџџџ:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates
Ш"
ж
while_body_301459
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_1_301483_0:P,
while_lstm_cell_1_301485_0:P(
while_lstm_cell_1_301487_0:P
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_1_301483:P*
while_lstm_cell_1_301485:P&
while_lstm_cell_1_301487:PЂ)while/lstm_cell_1/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ў
)while/lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1_301483_0while_lstm_cell_1_301485_0while_lstm_cell_1_301487_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_301445л
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity2while/lstm_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
while/Identity_5Identity2while/lstm_cell_1/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџx

while/NoOpNoOp*^while/lstm_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_1_301483while_lstm_cell_1_301483_0"6
while_lstm_cell_1_301485while_lstm_cell_1_301485_0"6
while_lstm_cell_1_301487while_lstm_cell_1_301487_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2V
)while/lstm_cell_1/StatefulPartitionedCall)while/lstm_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Е
У
while_cond_305249
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_305249___redundant_placeholder04
0while_while_cond_305249___redundant_placeholder14
0while_while_cond_305249___redundant_placeholder24
0while_while_cond_305249___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
уJ

B__inference_lstm_2_layer_call_and_return_conditional_losses_302979

inputs<
*lstm_cell_2_matmul_readvariableop_resource:(>
,lstm_cell_2_matmul_1_readvariableop_resource:
(9
+lstm_cell_2_biasadd_readvariableop_resource:(
identityЂ"lstm_cell_2/BiasAdd/ReadVariableOpЂ!lstm_cell_2/MatMul/ReadVariableOpЂ#lstm_cell_2/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype0
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :к
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitl
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
u
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
f
lstm_cell_2/ReluRelulstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
x
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
c
lstm_cell_2/Relu_1Relulstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0 lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_302894*
condR
while_cond_302893*K
output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Н
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Е
У
while_cond_302146
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_302146___redundant_placeholder04
0while_while_cond_302146___redundant_placeholder14
0while_while_cond_302146___redundant_placeholder24
0while_while_cond_302146___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
ц
Є
&__inference_model_layer_call_fn_303835

inputs
unknown:Ex
	unknown_0:x
	unknown_1:x
	unknown_2:P
	unknown_3:P
	unknown_4:P
	unknown_5:(
	unknown_6:
(
	unknown_7:(
	unknown_8:


	unknown_9:


unknown_10:
@

unknown_11:@

unknown_12:@ 

unknown_13: 

unknown_14:@ 

unknown_15: 

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:
identity

identity_1ЂStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_302686o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџE: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs
Ђ
Ч
model_lstm_2_while_cond_3008856
2model_lstm_2_while_model_lstm_2_while_loop_counter<
8model_lstm_2_while_model_lstm_2_while_maximum_iterations"
model_lstm_2_while_placeholder$
 model_lstm_2_while_placeholder_1$
 model_lstm_2_while_placeholder_2$
 model_lstm_2_while_placeholder_38
4model_lstm_2_while_less_model_lstm_2_strided_slice_1N
Jmodel_lstm_2_while_model_lstm_2_while_cond_300885___redundant_placeholder0N
Jmodel_lstm_2_while_model_lstm_2_while_cond_300885___redundant_placeholder1N
Jmodel_lstm_2_while_model_lstm_2_while_cond_300885___redundant_placeholder2N
Jmodel_lstm_2_while_model_lstm_2_while_cond_300885___redundant_placeholder3
model_lstm_2_while_identity

model/lstm_2/while/LessLessmodel_lstm_2_while_placeholder4model_lstm_2_while_less_model_lstm_2_strided_slice_1*
T0*
_output_shapes
: e
model/lstm_2/while/IdentityIdentitymodel/lstm_2/while/Less:z:0*
T0
*
_output_shapes
: "C
model_lstm_2_while_identity$model/lstm_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ
:џџџџџџџџџ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
љВ
Х
A__inference_model_layer_call_and_return_conditional_losses_304373

inputs?
-lstm_lstm_cell_matmul_readvariableop_resource:ExA
/lstm_lstm_cell_matmul_1_readvariableop_resource:x<
.lstm_lstm_cell_biasadd_readvariableop_resource:xC
1lstm_1_lstm_cell_1_matmul_readvariableop_resource:PE
3lstm_1_lstm_cell_1_matmul_1_readvariableop_resource:P@
2lstm_1_lstm_cell_1_biasadd_readvariableop_resource:PC
1lstm_2_lstm_cell_2_matmul_readvariableop_resource:(E
3lstm_2_lstm_cell_2_matmul_1_readvariableop_resource:
(@
2lstm_2_lstm_cell_2_biasadd_readvariableop_resource:(6
$dense_matmul_readvariableop_resource:

3
%dense_biasadd_readvariableop_resource:
<
*fingerprint_matmul_readvariableop_resource:
@9
+fingerprint_biasadd_readvariableop_resource:@8
&dense_3_matmul_readvariableop_resource:@ 5
'dense_3_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: 8
&dense_4_matmul_readvariableop_resource: 5
'dense_4_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource: 5
'dense_2_biasadd_readvariableop_resource:@
.identity_output_matmul_readvariableop_resource:=
/identity_output_biasadd_readvariableop_resource:=
+class_output_matmul_readvariableop_resource::
,class_output_biasadd_readvariableop_resource:
identity

identity_1Ђ#class_output/BiasAdd/ReadVariableOpЂ"class_output/MatMul/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpЂdense_3/BiasAdd/ReadVariableOpЂdense_3/MatMul/ReadVariableOpЂdense_4/BiasAdd/ReadVariableOpЂdense_4/MatMul/ReadVariableOpЂ"fingerprint/BiasAdd/ReadVariableOpЂ!fingerprint/MatMul/ReadVariableOpЂ&identity_output/BiasAdd/ReadVariableOpЂ%identity_output/MatMul/ReadVariableOpЂ%lstm/lstm_cell/BiasAdd/ReadVariableOpЂ$lstm/lstm_cell/MatMul/ReadVariableOpЂ&lstm/lstm_cell/MatMul_1/ReadVariableOpЂ
lstm/whileЂ)lstm_1/lstm_cell_1/BiasAdd/ReadVariableOpЂ(lstm_1/lstm_cell_1/MatMul/ReadVariableOpЂ*lstm_1/lstm_cell_1/MatMul_1/ReadVariableOpЂlstm_1/whileЂ)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOpЂ(lstm_2/lstm_cell_2/MatMul/ReadVariableOpЂ*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOpЂlstm_2/while@

lstm/ShapeShapeinputs*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџW
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџh
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџEN
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџУ
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   я
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвd
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџE*
shrink_axis_mask
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes

:Ex*
dtype0
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџx
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx`
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :у
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitr
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџt
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџl
lstm/lstm_cell/ReluRelulstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџt
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџi
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџs
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ч
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвK
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџY
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : П

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_lstm_cell_matmul_readvariableop_resource/lstm_lstm_cell_matmul_1_readvariableop_resource.lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *"
bodyR
lstm_while_body_303951*"
condR
lstm_while_cond_303950*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   б
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0m
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџf
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB: 
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ѕ
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ`
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
dropout/IdentityIdentitylstm/transpose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџU
lstm_1/ShapeShapedropout/Identity:output:0*
T0*
_output_shapes
:d
lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџY
lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_1/transpose	Transposedropout/Identity:output:0lstm_1/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:f
lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЩ
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ѕ
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвf
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
(lstm_1/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp1lstm_1_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0Ј
lstm_1/lstm_cell_1/MatMulMatMullstm_1/strided_slice_2:output:00lstm_1/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
*lstm_1/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp3lstm_1_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype0Ђ
lstm_1/lstm_cell_1/MatMul_1MatMullstm_1/zeros:output:02lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
lstm_1/lstm_cell_1/addAddV2#lstm_1/lstm_cell_1/MatMul:product:0%lstm_1/lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP
)lstm_1/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp2lstm_1_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0І
lstm_1/lstm_cell_1/BiasAddBiasAddlstm_1/lstm_cell_1/add:z:01lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPd
"lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
lstm_1/lstm_cell_1/splitSplit+lstm_1/lstm_cell_1/split/split_dim:output:0#lstm_1/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitz
lstm_1/lstm_cell_1/SigmoidSigmoid!lstm_1/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ|
lstm_1/lstm_cell_1/Sigmoid_1Sigmoid!lstm_1/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_1/lstm_cell_1/mulMul lstm_1/lstm_cell_1/Sigmoid_1:y:0lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџt
lstm_1/lstm_cell_1/ReluRelu!lstm_1/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_1/lstm_cell_1/mul_1Mullstm_1/lstm_cell_1/Sigmoid:y:0%lstm_1/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_1/lstm_cell_1/add_1AddV2lstm_1/lstm_cell_1/mul:z:0lstm_1/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ|
lstm_1/lstm_cell_1/Sigmoid_2Sigmoid!lstm_1/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџq
lstm_1/lstm_cell_1/Relu_1Relulstm_1/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_1/lstm_cell_1/mul_2Mul lstm_1/lstm_cell_1/Sigmoid_2:y:0'lstm_1/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџu
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Э
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвM
lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : с
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_1_lstm_cell_1_matmul_readvariableop_resource3lstm_1_lstm_cell_1_matmul_1_readvariableop_resource2lstm_1_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_1_while_body_304091*$
condR
lstm_1_while_cond_304090*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   з
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm_1/while:output:3@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0o
lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџh
lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskl
lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ћ
lstm_1/transpose_1	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџb
lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    R
lstm_2/ShapeShapelstm_1/transpose_1:y:0*
T0*
_output_shapes
:d
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
Y
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
j
lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_2/transpose	Transposelstm_1/transpose_1:y:0lstm_2/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
lstm_2/Shape_1Shapelstm_2/transpose:y:0*
T0*
_output_shapes
:f
lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
lstm_2/strided_slice_1StridedSlicelstm_2/Shape_1:output:0%lstm_2/strided_slice_1/stack:output:0'lstm_2/strided_slice_1/stack_1:output:0'lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЩ
lstm_2/TensorArrayV2TensorListReserve+lstm_2/TensorArrayV2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ѕ
.lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_2/transpose:y:0Elstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвf
lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_2/strided_slice_2StridedSlicelstm_2/transpose:y:0%lstm_2/strided_slice_2/stack:output:0'lstm_2/strided_slice_2/stack_1:output:0'lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
(lstm_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp1lstm_2_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0Ј
lstm_2/lstm_cell_2/MatMulMatMullstm_2/strided_slice_2:output:00lstm_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp3lstm_2_lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype0Ђ
lstm_2/lstm_cell_2/MatMul_1MatMullstm_2/zeros:output:02lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
lstm_2/lstm_cell_2/addAddV2#lstm_2/lstm_cell_2/MatMul:product:0%lstm_2/lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(
)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp2lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0І
lstm_2/lstm_cell_2/BiasAddBiasAddlstm_2/lstm_cell_2/add:z:01lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(d
"lstm_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
lstm_2/lstm_cell_2/splitSplit+lstm_2/lstm_cell_2/split/split_dim:output:0#lstm_2/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitz
lstm_2/lstm_cell_2/SigmoidSigmoid!lstm_2/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
|
lstm_2/lstm_cell_2/Sigmoid_1Sigmoid!lstm_2/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_2/lstm_cell_2/mulMul lstm_2/lstm_cell_2/Sigmoid_1:y:0lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
t
lstm_2/lstm_cell_2/ReluRelu!lstm_2/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_2/lstm_cell_2/mul_1Mullstm_2/lstm_cell_2/Sigmoid:y:0%lstm_2/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_2/lstm_cell_2/add_1AddV2lstm_2/lstm_cell_2/mul:z:0lstm_2/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
|
lstm_2/lstm_cell_2/Sigmoid_2Sigmoid!lstm_2/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_2/lstm_cell_2/Relu_1Relulstm_2/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_2/lstm_cell_2/mul_2Mul lstm_2/lstm_cell_2/Sigmoid_2:y:0'lstm_2/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
u
$lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   e
#lstm_2/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :к
lstm_2/TensorArrayV2_1TensorListReserve-lstm_2/TensorArrayV2_1/element_shape:output:0,lstm_2/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвM
lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : с
lstm_2/whileWhile"lstm_2/while/loop_counter:output:0(lstm_2/while/maximum_iterations:output:0lstm_2/time:output:0lstm_2/TensorArrayV2_1:handle:0lstm_2/zeros:output:0lstm_2/zeros_1:output:0lstm_2/strided_slice_1:output:0>lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_2_lstm_cell_2_matmul_readvariableop_resource3lstm_2_lstm_cell_2_matmul_1_readvariableop_resource2lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_2_while_body_304231*$
condR
lstm_2_while_cond_304230*K
output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *
parallel_iterations 
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ы
)lstm_2/TensorArrayV2Stack/TensorListStackTensorListStacklstm_2/while:output:3@lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementso
lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџh
lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
lstm_2/strided_slice_3StridedSlice2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_2/strided_slice_3/stack:output:0'lstm_2/strided_slice_3/stack_1:output:0'lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maskl
lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ћ
lstm_2/transpose_1	Transpose2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
b
lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0
dense/MatMulMatMullstm_2/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ

!fingerprint/MatMul/ReadVariableOpReadVariableOp*fingerprint_matmul_readvariableop_resource*
_output_shapes

:
@*
dtype0
fingerprint/MatMulMatMuldense/Relu:activations:0)fingerprint/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
"fingerprint/BiasAdd/ReadVariableOpReadVariableOp+fingerprint_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
fingerprint/BiasAddBiasAddfingerprint/MatMul:product:0*fingerprint/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@h
fingerprint/ReluRelufingerprint/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
dense_3/MatMulMatMulfingerprint/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
dense_1/MatMulMatMulfingerprint/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
%identity_output/MatMul/ReadVariableOpReadVariableOp.identity_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
identity_output/MatMulMatMuldense_4/Relu:activations:0-identity_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&identity_output/BiasAdd/ReadVariableOpReadVariableOp/identity_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0І
identity_output/BiasAddBiasAdd identity_output/MatMul:product:0.identity_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџv
identity_output/SoftmaxSoftmax identity_output/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
"class_output/MatMul/ReadVariableOpReadVariableOp+class_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
class_output/MatMulMatMuldense_2/Relu:activations:0*class_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
#class_output/BiasAdd/ReadVariableOpReadVariableOp,class_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
class_output/BiasAddBiasAddclass_output/MatMul:product:0+class_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџp
class_output/SoftmaxSoftmaxclass_output/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџm
IdentityIdentityclass_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџr

Identity_1Identity!identity_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp$^class_output/BiasAdd/ReadVariableOp#^class_output/MatMul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp#^fingerprint/BiasAdd/ReadVariableOp"^fingerprint/MatMul/ReadVariableOp'^identity_output/BiasAdd/ReadVariableOp&^identity_output/MatMul/ReadVariableOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while*^lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp)^lstm_1/lstm_cell_1/MatMul/ReadVariableOp+^lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp^lstm_1/while*^lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp)^lstm_2/lstm_cell_2/MatMul/ReadVariableOp+^lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp^lstm_2/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџE: : : : : : : : : : : : : : : : : : : : : : : : : 2J
#class_output/BiasAdd/ReadVariableOp#class_output/BiasAdd/ReadVariableOp2H
"class_output/MatMul/ReadVariableOp"class_output/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2H
"fingerprint/BiasAdd/ReadVariableOp"fingerprint/BiasAdd/ReadVariableOp2F
!fingerprint/MatMul/ReadVariableOp!fingerprint/MatMul/ReadVariableOp2P
&identity_output/BiasAdd/ReadVariableOp&identity_output/BiasAdd/ReadVariableOp2N
%identity_output/MatMul/ReadVariableOp%identity_output/MatMul/ReadVariableOp2N
%lstm/lstm_cell/BiasAdd/ReadVariableOp%lstm/lstm_cell/BiasAdd/ReadVariableOp2L
$lstm/lstm_cell/MatMul/ReadVariableOp$lstm/lstm_cell/MatMul/ReadVariableOp2P
&lstm/lstm_cell/MatMul_1/ReadVariableOp&lstm/lstm_cell/MatMul_1/ReadVariableOp2

lstm/while
lstm/while2V
)lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp)lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp2T
(lstm_1/lstm_cell_1/MatMul/ReadVariableOp(lstm_1/lstm_cell_1/MatMul/ReadVariableOp2X
*lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp*lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp2
lstm_1/whilelstm_1/while2V
)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp2T
(lstm_2/lstm_cell_2/MatMul/ReadVariableOp(lstm_2/lstm_cell_2/MatMul/ReadVariableOp2X
*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp2
lstm_2/whilelstm_2/while:S O
+
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs
Е
У
while_cond_306513
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_306513___redundant_placeholder04
0while_while_cond_306513___redundant_placeholder14
0while_while_cond_306513___redundant_placeholder24
0while_while_cond_306513___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ
:џџџџџџџџџ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
ђ
Б
'__inference_lstm_2_layer_call_fn_306153

inputs
unknown:(
	unknown_0:
(
	unknown_1:(
identityЂStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_302540o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
КЈ
ђ&
__inference__traced_save_307492
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop1
-savev2_fingerprint_kernel_read_readvariableop/
+savev2_fingerprint_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop2
.savev2_class_output_kernel_read_readvariableop0
,savev2_class_output_bias_read_readvariableop5
1savev2_identity_output_kernel_read_readvariableop3
/savev2_identity_output_bias_read_readvariableop4
0savev2_lstm_lstm_cell_kernel_read_readvariableop>
:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop2
.savev2_lstm_lstm_cell_bias_read_readvariableop8
4savev2_lstm_1_lstm_cell_1_kernel_read_readvariableopB
>savev2_lstm_1_lstm_cell_1_recurrent_kernel_read_readvariableop6
2savev2_lstm_1_lstm_cell_1_bias_read_readvariableop8
4savev2_lstm_2_lstm_cell_2_kernel_read_readvariableopB
>savev2_lstm_2_lstm_cell_2_recurrent_kernel_read_readvariableop6
2savev2_lstm_2_lstm_cell_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop8
4savev2_adam_fingerprint_kernel_m_read_readvariableop6
2savev2_adam_fingerprint_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop9
5savev2_adam_class_output_kernel_m_read_readvariableop7
3savev2_adam_class_output_bias_m_read_readvariableop<
8savev2_adam_identity_output_kernel_m_read_readvariableop:
6savev2_adam_identity_output_bias_m_read_readvariableop;
7savev2_adam_lstm_lstm_cell_kernel_m_read_readvariableopE
Asavev2_adam_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop9
5savev2_adam_lstm_lstm_cell_bias_m_read_readvariableop?
;savev2_adam_lstm_1_lstm_cell_1_kernel_m_read_readvariableopI
Esavev2_adam_lstm_1_lstm_cell_1_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_1_lstm_cell_1_bias_m_read_readvariableop?
;savev2_adam_lstm_2_lstm_cell_2_kernel_m_read_readvariableopI
Esavev2_adam_lstm_2_lstm_cell_2_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_2_lstm_cell_2_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop8
4savev2_adam_fingerprint_kernel_v_read_readvariableop6
2savev2_adam_fingerprint_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop9
5savev2_adam_class_output_kernel_v_read_readvariableop7
3savev2_adam_class_output_bias_v_read_readvariableop<
8savev2_adam_identity_output_kernel_v_read_readvariableop:
6savev2_adam_identity_output_bias_v_read_readvariableop;
7savev2_adam_lstm_lstm_cell_kernel_v_read_readvariableopE
Asavev2_adam_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop9
5savev2_adam_lstm_lstm_cell_bias_v_read_readvariableop?
;savev2_adam_lstm_1_lstm_cell_1_kernel_v_read_readvariableopI
Esavev2_adam_lstm_1_lstm_cell_1_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_1_lstm_cell_1_bias_v_read_readvariableop?
;savev2_adam_lstm_2_lstm_cell_2_kernel_v_read_readvariableopI
Esavev2_adam_lstm_2_lstm_cell_2_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_2_lstm_cell_2_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: /
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:[*
dtype0*К.
valueА.B­.[B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHІ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:[*
dtype0*Ы
valueСBО[B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ј%
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop-savev2_fingerprint_kernel_read_readvariableop+savev2_fingerprint_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop.savev2_class_output_kernel_read_readvariableop,savev2_class_output_bias_read_readvariableop1savev2_identity_output_kernel_read_readvariableop/savev2_identity_output_bias_read_readvariableop0savev2_lstm_lstm_cell_kernel_read_readvariableop:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop.savev2_lstm_lstm_cell_bias_read_readvariableop4savev2_lstm_1_lstm_cell_1_kernel_read_readvariableop>savev2_lstm_1_lstm_cell_1_recurrent_kernel_read_readvariableop2savev2_lstm_1_lstm_cell_1_bias_read_readvariableop4savev2_lstm_2_lstm_cell_2_kernel_read_readvariableop>savev2_lstm_2_lstm_cell_2_recurrent_kernel_read_readvariableop2savev2_lstm_2_lstm_cell_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop4savev2_adam_fingerprint_kernel_m_read_readvariableop2savev2_adam_fingerprint_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop5savev2_adam_class_output_kernel_m_read_readvariableop3savev2_adam_class_output_bias_m_read_readvariableop8savev2_adam_identity_output_kernel_m_read_readvariableop6savev2_adam_identity_output_bias_m_read_readvariableop7savev2_adam_lstm_lstm_cell_kernel_m_read_readvariableopAsavev2_adam_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop5savev2_adam_lstm_lstm_cell_bias_m_read_readvariableop;savev2_adam_lstm_1_lstm_cell_1_kernel_m_read_readvariableopEsavev2_adam_lstm_1_lstm_cell_1_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_1_lstm_cell_1_bias_m_read_readvariableop;savev2_adam_lstm_2_lstm_cell_2_kernel_m_read_readvariableopEsavev2_adam_lstm_2_lstm_cell_2_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_2_lstm_cell_2_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop4savev2_adam_fingerprint_kernel_v_read_readvariableop2savev2_adam_fingerprint_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop5savev2_adam_class_output_kernel_v_read_readvariableop3savev2_adam_class_output_bias_v_read_readvariableop8savev2_adam_identity_output_kernel_v_read_readvariableop6savev2_adam_identity_output_bias_v_read_readvariableop7savev2_adam_lstm_lstm_cell_kernel_v_read_readvariableopAsavev2_adam_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop5savev2_adam_lstm_lstm_cell_bias_v_read_readvariableop;savev2_adam_lstm_1_lstm_cell_1_kernel_v_read_readvariableopEsavev2_adam_lstm_1_lstm_cell_1_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_1_lstm_cell_1_bias_v_read_readvariableop;savev2_adam_lstm_2_lstm_cell_2_kernel_v_read_readvariableopEsavev2_adam_lstm_2_lstm_cell_2_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_2_lstm_cell_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *i
dtypes_
]2[	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ё
_input_shapes
: :

:
:
@:@:@ : :@ : : :: ::::::Ex:x:x:P:P:P:(:
(:(: : : : : : : : : : : : : : : :

:
:
@:@:@ : :@ : : :: ::::::Ex:x:x:P:P:P:(:
(:(:

:
:
@:@:@ : :@ : : :: ::::::Ex:x:x:P:P:P:(:
(:(: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

:@ : 

_output_shapes
: :$	 

_output_shapes

: : 


_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:Ex:$ 

_output_shapes

:x: 

_output_shapes
:x:$ 

_output_shapes

:P:$ 

_output_shapes

:P: 

_output_shapes
:P:$ 

_output_shapes

:(:$ 

_output_shapes

:
(: 

_output_shapes
:(:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :$) 

_output_shapes

:

: *

_output_shapes
:
:$+ 

_output_shapes

:
@: ,

_output_shapes
:@:$- 

_output_shapes

:@ : .

_output_shapes
: :$/ 

_output_shapes

:@ : 0

_output_shapes
: :$1 

_output_shapes

: : 2

_output_shapes
::$3 

_output_shapes

: : 4

_output_shapes
::$5 

_output_shapes

:: 6

_output_shapes
::$7 

_output_shapes

:: 8

_output_shapes
::$9 

_output_shapes

:Ex:$: 

_output_shapes

:x: ;

_output_shapes
:x:$< 

_output_shapes

:P:$= 

_output_shapes

:P: >

_output_shapes
:P:$? 

_output_shapes

:(:$@ 

_output_shapes

:
(: A

_output_shapes
:(:$B 

_output_shapes

:

: C

_output_shapes
:
:$D 

_output_shapes

:
@: E

_output_shapes
:@:$F 

_output_shapes

:@ : G

_output_shapes
: :$H 

_output_shapes

:@ : I

_output_shapes
: :$J 

_output_shapes

: : K

_output_shapes
::$L 

_output_shapes

: : M

_output_shapes
::$N 

_output_shapes

:: O

_output_shapes
::$P 

_output_shapes

:: Q

_output_shapes
::$R 

_output_shapes

:Ex:$S 

_output_shapes

:x: T

_output_shapes
:x:$U 

_output_shapes

:P:$V 

_output_shapes

:P: W

_output_shapes
:P:$X 

_output_shapes

:(:$Y 

_output_shapes

:
(: Z

_output_shapes
:(:[

_output_shapes
: 
Є
Г
'__inference_lstm_1_layer_call_fn_305515
inputs_0
unknown:P
	unknown_0:P
	unknown_1:P
identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_301528|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
љ	
Я
lstm_2_while_cond_304718*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3,
(lstm_2_while_less_lstm_2_strided_slice_1B
>lstm_2_while_lstm_2_while_cond_304718___redundant_placeholder0B
>lstm_2_while_lstm_2_while_cond_304718___redundant_placeholder1B
>lstm_2_while_lstm_2_while_cond_304718___redundant_placeholder2B
>lstm_2_while_lstm_2_while_cond_304718___redundant_placeholder3
lstm_2_while_identity
~
lstm_2/while/LessLesslstm_2_while_placeholder(lstm_2_while_less_lstm_2_strided_slice_1*
T0*
_output_shapes
: Y
lstm_2/while/IdentityIdentitylstm_2/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_2_while_identitylstm_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ
:џџџџџџџџџ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
нI

B__inference_lstm_1_layer_call_and_return_conditional_losses_305977

inputs<
*lstm_cell_1_matmul_readvariableop_resource:P>
,lstm_cell_1_matmul_1_readvariableop_resource:P9
+lstm_cell_1_biasadd_readvariableop_resource:P
identityЂ"lstm_cell_1/BiasAdd/ReadVariableOpЂ!lstm_cell_1/MatMul/ReadVariableOpЂ#lstm_cell_1/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype0
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :к
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitl
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџn
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџu
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
lstm_cell_1/ReluRelulstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџx
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџc
lstm_cell_1/Relu_1Relulstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0 lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_1_matmul_readvariableop_resource,lstm_cell_1_matmul_1_readvariableop_resource+lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_305893*
condR
while_cond_305892*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Т
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџН
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Г
'__inference_lstm_2_layer_call_fn_306131
inputs_0
unknown:(
	unknown_0:
(
	unknown_1:(
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_301880o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0


b
C__inference_dropout_layer_call_and_return_conditional_losses_305504

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Њ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџs
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ц
ђ
,__inference_lstm_cell_1_layer_call_fn_307036

inputs
states_0
states_1
unknown:P
	unknown_0:P
	unknown_1:P
identity

identity_1

identity_2ЂStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_301591o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/1
ц
Є
&__inference_model_layer_call_fn_303892

inputs
unknown:Ex
	unknown_0:x
	unknown_1:x
	unknown_2:P
	unknown_3:P
	unknown_4:P
	unknown_5:(
	unknown_6:
(
	unknown_7:(
	unknown_8:


	unknown_9:


unknown_10:
@

unknown_11:@

unknown_12:@ 

unknown_13: 

unknown_14:@ 

unknown_15: 

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:
identity

identity_1ЂStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_303467o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџE: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs
ъ


model_lstm_while_cond_3006052
.model_lstm_while_model_lstm_while_loop_counter8
4model_lstm_while_model_lstm_while_maximum_iterations 
model_lstm_while_placeholder"
model_lstm_while_placeholder_1"
model_lstm_while_placeholder_2"
model_lstm_while_placeholder_34
0model_lstm_while_less_model_lstm_strided_slice_1J
Fmodel_lstm_while_model_lstm_while_cond_300605___redundant_placeholder0J
Fmodel_lstm_while_model_lstm_while_cond_300605___redundant_placeholder1J
Fmodel_lstm_while_model_lstm_while_cond_300605___redundant_placeholder2J
Fmodel_lstm_while_model_lstm_while_cond_300605___redundant_placeholder3
model_lstm_while_identity

model/lstm/while/LessLessmodel_lstm_while_placeholder0model_lstm_while_less_model_lstm_strided_slice_1*
T0*
_output_shapes
: a
model/lstm/while/IdentityIdentitymodel/lstm/while/Less:z:0*
T0
*
_output_shapes
: "?
model_lstm_while_identity"model/lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
 
Б
%__inference_lstm_layer_call_fn_304883
inputs_0
unknown:Ex
	unknown_0:x
	unknown_1:x
identityЂStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_301369|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџE: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџE
"
_user_specified_name
inputs/0
"
Ш
while_body_301300
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_301324_0:Ex*
while_lstm_cell_301326_0:x&
while_lstm_cell_301328_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_301324:Ex(
while_lstm_cell_301326:x$
while_lstm_cell_301328:xЂ'while/lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџE*
element_dtype0Є
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_301324_0while_lstm_cell_301326_0while_lstm_cell_301328_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_301241й
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџv

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"2
while_lstm_cell_301324while_lstm_cell_301324_0"2
while_lstm_cell_301326while_lstm_cell_301326_0"2
while_lstm_cell_301328while_lstm_cell_301328_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
ЊК
Х
A__inference_model_layer_call_and_return_conditional_losses_304861

inputs?
-lstm_lstm_cell_matmul_readvariableop_resource:ExA
/lstm_lstm_cell_matmul_1_readvariableop_resource:x<
.lstm_lstm_cell_biasadd_readvariableop_resource:xC
1lstm_1_lstm_cell_1_matmul_readvariableop_resource:PE
3lstm_1_lstm_cell_1_matmul_1_readvariableop_resource:P@
2lstm_1_lstm_cell_1_biasadd_readvariableop_resource:PC
1lstm_2_lstm_cell_2_matmul_readvariableop_resource:(E
3lstm_2_lstm_cell_2_matmul_1_readvariableop_resource:
(@
2lstm_2_lstm_cell_2_biasadd_readvariableop_resource:(6
$dense_matmul_readvariableop_resource:

3
%dense_biasadd_readvariableop_resource:
<
*fingerprint_matmul_readvariableop_resource:
@9
+fingerprint_biasadd_readvariableop_resource:@8
&dense_3_matmul_readvariableop_resource:@ 5
'dense_3_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: 8
&dense_4_matmul_readvariableop_resource: 5
'dense_4_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource: 5
'dense_2_biasadd_readvariableop_resource:@
.identity_output_matmul_readvariableop_resource:=
/identity_output_biasadd_readvariableop_resource:=
+class_output_matmul_readvariableop_resource::
,class_output_biasadd_readvariableop_resource:
identity

identity_1Ђ#class_output/BiasAdd/ReadVariableOpЂ"class_output/MatMul/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpЂdense_3/BiasAdd/ReadVariableOpЂdense_3/MatMul/ReadVariableOpЂdense_4/BiasAdd/ReadVariableOpЂdense_4/MatMul/ReadVariableOpЂ"fingerprint/BiasAdd/ReadVariableOpЂ!fingerprint/MatMul/ReadVariableOpЂ&identity_output/BiasAdd/ReadVariableOpЂ%identity_output/MatMul/ReadVariableOpЂ%lstm/lstm_cell/BiasAdd/ReadVariableOpЂ$lstm/lstm_cell/MatMul/ReadVariableOpЂ&lstm/lstm_cell/MatMul_1/ReadVariableOpЂ
lstm/whileЂ)lstm_1/lstm_cell_1/BiasAdd/ReadVariableOpЂ(lstm_1/lstm_cell_1/MatMul/ReadVariableOpЂ*lstm_1/lstm_cell_1/MatMul_1/ReadVariableOpЂlstm_1/whileЂ)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOpЂ(lstm_2/lstm_cell_2/MatMul/ReadVariableOpЂ*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOpЂlstm_2/while@

lstm/ShapeShapeinputs*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџW
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџh
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџEN
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџУ
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   я
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвd
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџE*
shrink_axis_mask
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes

:Ex*
dtype0
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџx
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx`
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :у
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitr
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџt
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџl
lstm/lstm_cell/ReluRelulstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџt
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџi
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџs
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ч
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвK
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџY
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : П

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_lstm_cell_matmul_readvariableop_resource/lstm_lstm_cell_matmul_1_readvariableop_resource.lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *"
bodyR
lstm_while_body_304432*"
condR
lstm_while_cond_304431*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   б
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0m
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџf
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB: 
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ѕ
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ`
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?
dropout/dropout/MulMullstm/transpose_1:y:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџY
dropout/dropout/ShapeShapelstm/transpose_1:y:0*
T0*
_output_shapes
: 
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Т
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџU
lstm_1/ShapeShapedropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:d
lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџY
lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_1/transpose	Transposedropout/dropout/Mul_1:z:0lstm_1/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:f
lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЩ
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ѕ
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвf
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
(lstm_1/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp1lstm_1_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0Ј
lstm_1/lstm_cell_1/MatMulMatMullstm_1/strided_slice_2:output:00lstm_1/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
*lstm_1/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp3lstm_1_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype0Ђ
lstm_1/lstm_cell_1/MatMul_1MatMullstm_1/zeros:output:02lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
lstm_1/lstm_cell_1/addAddV2#lstm_1/lstm_cell_1/MatMul:product:0%lstm_1/lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP
)lstm_1/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp2lstm_1_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0І
lstm_1/lstm_cell_1/BiasAddBiasAddlstm_1/lstm_cell_1/add:z:01lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPd
"lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
lstm_1/lstm_cell_1/splitSplit+lstm_1/lstm_cell_1/split/split_dim:output:0#lstm_1/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitz
lstm_1/lstm_cell_1/SigmoidSigmoid!lstm_1/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ|
lstm_1/lstm_cell_1/Sigmoid_1Sigmoid!lstm_1/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_1/lstm_cell_1/mulMul lstm_1/lstm_cell_1/Sigmoid_1:y:0lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџt
lstm_1/lstm_cell_1/ReluRelu!lstm_1/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_1/lstm_cell_1/mul_1Mullstm_1/lstm_cell_1/Sigmoid:y:0%lstm_1/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_1/lstm_cell_1/add_1AddV2lstm_1/lstm_cell_1/mul:z:0lstm_1/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ|
lstm_1/lstm_cell_1/Sigmoid_2Sigmoid!lstm_1/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџq
lstm_1/lstm_cell_1/Relu_1Relulstm_1/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_1/lstm_cell_1/mul_2Mul lstm_1/lstm_cell_1/Sigmoid_2:y:0'lstm_1/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџu
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Э
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвM
lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : с
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_1_lstm_cell_1_matmul_readvariableop_resource3lstm_1_lstm_cell_1_matmul_1_readvariableop_resource2lstm_1_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_1_while_body_304579*$
condR
lstm_1_while_cond_304578*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   з
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm_1/while:output:3@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0o
lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџh
lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskl
lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ћ
lstm_1/transpose_1	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџb
lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    R
lstm_2/ShapeShapelstm_1/transpose_1:y:0*
T0*
_output_shapes
:d
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
Y
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
j
lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_2/transpose	Transposelstm_1/transpose_1:y:0lstm_2/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
lstm_2/Shape_1Shapelstm_2/transpose:y:0*
T0*
_output_shapes
:f
lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
lstm_2/strided_slice_1StridedSlicelstm_2/Shape_1:output:0%lstm_2/strided_slice_1/stack:output:0'lstm_2/strided_slice_1/stack_1:output:0'lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЩ
lstm_2/TensorArrayV2TensorListReserve+lstm_2/TensorArrayV2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ѕ
.lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_2/transpose:y:0Elstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвf
lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_2/strided_slice_2StridedSlicelstm_2/transpose:y:0%lstm_2/strided_slice_2/stack:output:0'lstm_2/strided_slice_2/stack_1:output:0'lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
(lstm_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp1lstm_2_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0Ј
lstm_2/lstm_cell_2/MatMulMatMullstm_2/strided_slice_2:output:00lstm_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp3lstm_2_lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype0Ђ
lstm_2/lstm_cell_2/MatMul_1MatMullstm_2/zeros:output:02lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
lstm_2/lstm_cell_2/addAddV2#lstm_2/lstm_cell_2/MatMul:product:0%lstm_2/lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(
)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp2lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0І
lstm_2/lstm_cell_2/BiasAddBiasAddlstm_2/lstm_cell_2/add:z:01lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(d
"lstm_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
lstm_2/lstm_cell_2/splitSplit+lstm_2/lstm_cell_2/split/split_dim:output:0#lstm_2/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitz
lstm_2/lstm_cell_2/SigmoidSigmoid!lstm_2/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
|
lstm_2/lstm_cell_2/Sigmoid_1Sigmoid!lstm_2/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_2/lstm_cell_2/mulMul lstm_2/lstm_cell_2/Sigmoid_1:y:0lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
t
lstm_2/lstm_cell_2/ReluRelu!lstm_2/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_2/lstm_cell_2/mul_1Mullstm_2/lstm_cell_2/Sigmoid:y:0%lstm_2/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_2/lstm_cell_2/add_1AddV2lstm_2/lstm_cell_2/mul:z:0lstm_2/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
|
lstm_2/lstm_cell_2/Sigmoid_2Sigmoid!lstm_2/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_2/lstm_cell_2/Relu_1Relulstm_2/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_2/lstm_cell_2/mul_2Mul lstm_2/lstm_cell_2/Sigmoid_2:y:0'lstm_2/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
u
$lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   e
#lstm_2/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :к
lstm_2/TensorArrayV2_1TensorListReserve-lstm_2/TensorArrayV2_1/element_shape:output:0,lstm_2/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвM
lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : с
lstm_2/whileWhile"lstm_2/while/loop_counter:output:0(lstm_2/while/maximum_iterations:output:0lstm_2/time:output:0lstm_2/TensorArrayV2_1:handle:0lstm_2/zeros:output:0lstm_2/zeros_1:output:0lstm_2/strided_slice_1:output:0>lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_2_lstm_cell_2_matmul_readvariableop_resource3lstm_2_lstm_cell_2_matmul_1_readvariableop_resource2lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_2_while_body_304719*$
condR
lstm_2_while_cond_304718*K
output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *
parallel_iterations 
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ы
)lstm_2/TensorArrayV2Stack/TensorListStackTensorListStacklstm_2/while:output:3@lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementso
lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџh
lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
lstm_2/strided_slice_3StridedSlice2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_2/strided_slice_3/stack:output:0'lstm_2/strided_slice_3/stack_1:output:0'lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maskl
lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ћ
lstm_2/transpose_1	Transpose2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
b
lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0
dense/MatMulMatMullstm_2/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ

!fingerprint/MatMul/ReadVariableOpReadVariableOp*fingerprint_matmul_readvariableop_resource*
_output_shapes

:
@*
dtype0
fingerprint/MatMulMatMuldense/Relu:activations:0)fingerprint/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
"fingerprint/BiasAdd/ReadVariableOpReadVariableOp+fingerprint_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
fingerprint/BiasAddBiasAddfingerprint/MatMul:product:0*fingerprint/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@h
fingerprint/ReluRelufingerprint/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
dense_3/MatMulMatMulfingerprint/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
dense_1/MatMulMatMulfingerprint/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
%identity_output/MatMul/ReadVariableOpReadVariableOp.identity_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
identity_output/MatMulMatMuldense_4/Relu:activations:0-identity_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&identity_output/BiasAdd/ReadVariableOpReadVariableOp/identity_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0І
identity_output/BiasAddBiasAdd identity_output/MatMul:product:0.identity_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџv
identity_output/SoftmaxSoftmax identity_output/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
"class_output/MatMul/ReadVariableOpReadVariableOp+class_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
class_output/MatMulMatMuldense_2/Relu:activations:0*class_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
#class_output/BiasAdd/ReadVariableOpReadVariableOp,class_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
class_output/BiasAddBiasAddclass_output/MatMul:product:0+class_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџp
class_output/SoftmaxSoftmaxclass_output/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџm
IdentityIdentityclass_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџr

Identity_1Identity!identity_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp$^class_output/BiasAdd/ReadVariableOp#^class_output/MatMul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp#^fingerprint/BiasAdd/ReadVariableOp"^fingerprint/MatMul/ReadVariableOp'^identity_output/BiasAdd/ReadVariableOp&^identity_output/MatMul/ReadVariableOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while*^lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp)^lstm_1/lstm_cell_1/MatMul/ReadVariableOp+^lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp^lstm_1/while*^lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp)^lstm_2/lstm_cell_2/MatMul/ReadVariableOp+^lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp^lstm_2/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџE: : : : : : : : : : : : : : : : : : : : : : : : : 2J
#class_output/BiasAdd/ReadVariableOp#class_output/BiasAdd/ReadVariableOp2H
"class_output/MatMul/ReadVariableOp"class_output/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2H
"fingerprint/BiasAdd/ReadVariableOp"fingerprint/BiasAdd/ReadVariableOp2F
!fingerprint/MatMul/ReadVariableOp!fingerprint/MatMul/ReadVariableOp2P
&identity_output/BiasAdd/ReadVariableOp&identity_output/BiasAdd/ReadVariableOp2N
%identity_output/MatMul/ReadVariableOp%identity_output/MatMul/ReadVariableOp2N
%lstm/lstm_cell/BiasAdd/ReadVariableOp%lstm/lstm_cell/BiasAdd/ReadVariableOp2L
$lstm/lstm_cell/MatMul/ReadVariableOp$lstm/lstm_cell/MatMul/ReadVariableOp2P
&lstm/lstm_cell/MatMul_1/ReadVariableOp&lstm/lstm_cell/MatMul_1/ReadVariableOp2

lstm/while
lstm/while2V
)lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp)lstm_1/lstm_cell_1/BiasAdd/ReadVariableOp2T
(lstm_1/lstm_cell_1/MatMul/ReadVariableOp(lstm_1/lstm_cell_1/MatMul/ReadVariableOp2X
*lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp*lstm_1/lstm_cell_1/MatMul_1/ReadVariableOp2
lstm_1/whilelstm_1/while2V
)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp2T
(lstm_2/lstm_cell_2/MatMul/ReadVariableOp(lstm_2/lstm_cell_2/MatMul/ReadVariableOp2X
*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp2
lstm_2/whilelstm_2/while:S O
+
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs
з8
С
while_body_306514
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_2_matmul_readvariableop_resource_0:(F
4while_lstm_cell_2_matmul_1_readvariableop_resource_0:
(A
3while_lstm_cell_2_biasadd_readvariableop_resource_0:(
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_2_matmul_readvariableop_resource:(D
2while_lstm_cell_2_matmul_1_readvariableop_resource:
(?
1while_lstm_cell_2_biasadd_readvariableop_resource:(Ђ(while/lstm_cell_2/BiasAdd/ReadVariableOpЂ'while/lstm_cell_2/MatMul/ReadVariableOpЂ)while/lstm_cell_2/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:(*
dtype0З
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype0
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype0Ѓ
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(c
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ь
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitx
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
z
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ
r
while/lstm_cell_2/ReluRelu while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0$while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
z
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
o
while/lstm_cell_2/Relu_1Reluwhile/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0&while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
x
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Э

while/NoOpNoOp)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : 2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
ф#
ж
while_body_302003
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_2_302027_0:(,
while_lstm_cell_2_302029_0:
((
while_lstm_cell_2_302031_0:(
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_2_302027:(*
while_lstm_cell_2_302029:
(&
while_lstm_cell_2_302031:(Ђ)while/lstm_cell_2/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ў
)while/lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_2_302027_0while_lstm_cell_2_302029_0while_lstm_cell_2_302031_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_301943r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:02while/lstm_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity2while/lstm_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

while/Identity_5Identity2while/lstm_cell_2/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
x

while/NoOpNoOp*^while/lstm_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_2_302027while_lstm_cell_2_302027_0"6
while_lstm_cell_2_302029while_lstm_cell_2_302029_0"6
while_lstm_cell_2_302031while_lstm_cell_2_302031_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : 2V
)while/lstm_cell_2/StatefulPartitionedCall)while/lstm_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
з8
С
while_body_306659
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_2_matmul_readvariableop_resource_0:(F
4while_lstm_cell_2_matmul_1_readvariableop_resource_0:
(A
3while_lstm_cell_2_biasadd_readvariableop_resource_0:(
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_2_matmul_readvariableop_resource:(D
2while_lstm_cell_2_matmul_1_readvariableop_resource:
(?
1while_lstm_cell_2_biasadd_readvariableop_resource:(Ђ(while/lstm_cell_2/BiasAdd/ReadVariableOpЂ'while/lstm_cell_2/MatMul/ReadVariableOpЂ)while/lstm_cell_2/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:(*
dtype0З
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype0
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype0Ѓ
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(c
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ь
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitx
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
z
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ
r
while/lstm_cell_2/ReluRelu while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0$while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
z
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
o
while/lstm_cell_2/Relu_1Reluwhile/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0&while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
x
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Э

while/NoOpNoOp)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : 2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
Е
У
while_cond_301809
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_301809___redundant_placeholder04
0while_while_cond_301809___redundant_placeholder14
0while_while_cond_301809___redundant_placeholder24
0while_while_cond_301809___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ
:џџџџџџџџџ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
з8
С
while_body_306369
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_2_matmul_readvariableop_resource_0:(F
4while_lstm_cell_2_matmul_1_readvariableop_resource_0:
(A
3while_lstm_cell_2_biasadd_readvariableop_resource_0:(
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_2_matmul_readvariableop_resource:(D
2while_lstm_cell_2_matmul_1_readvariableop_resource:
(?
1while_lstm_cell_2_biasadd_readvariableop_resource:(Ђ(while/lstm_cell_2/BiasAdd/ReadVariableOpЂ'while/lstm_cell_2/MatMul/ReadVariableOpЂ)while/lstm_cell_2/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:(*
dtype0З
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype0
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype0Ѓ
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(c
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ь
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitx
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
z
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ
r
while/lstm_cell_2/ReluRelu while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0$while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
z
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
o
while/lstm_cell_2/Relu_1Reluwhile/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0&while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
x
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Э

while/NoOpNoOp)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : 2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
Ш
ў
E__inference_lstm_cell_layer_call_and_return_conditional_losses_301095

inputs

states
states_10
matmul_readvariableop_resource:Ex2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Ex*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџN
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџE:џџџџџџџџџ:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates
уJ

B__inference_lstm_2_layer_call_and_return_conditional_losses_306744

inputs<
*lstm_cell_2_matmul_readvariableop_resource:(>
,lstm_cell_2_matmul_1_readvariableop_resource:
(9
+lstm_cell_2_biasadd_readvariableop_resource:(
identityЂ"lstm_cell_2/BiasAdd/ReadVariableOpЂ!lstm_cell_2/MatMul/ReadVariableOpЂ#lstm_cell_2/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype0
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :к
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitl
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
u
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
f
lstm_cell_2/ReluRelulstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
x
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
c
lstm_cell_2/Relu_1Relulstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0 lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_306659*
condR
while_cond_306658*K
output_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Н
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


ј
G__inference_fingerprint_layer_call_and_return_conditional_losses_302576

inputs0
matmul_readvariableop_resource:
@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
Е
У
while_cond_305392
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_305392___redundant_placeholder04
0while_while_cond_305392___redundant_placeholder14
0while_while_cond_305392___redundant_placeholder24
0while_while_cond_305392___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
і
Џ
%__inference_lstm_layer_call_fn_304894

inputs
unknown:Ex
	unknown_0:x
	unknown_1:x
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_302231s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџE: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs
џ7
§
B__inference_lstm_1_layer_call_and_return_conditional_losses_301528

inputs$
lstm_cell_1_301446:P$
lstm_cell_1_301448:P 
lstm_cell_1_301450:P
identityЂ#lstm_cell_1/StatefulPartitionedCallЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask№
#lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1_301446lstm_cell_1_301448lstm_cell_1_301450*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_301445n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Д
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1_301446lstm_cell_1_301448lstm_cell_1_301450*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_301459*
condR
while_cond_301458*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџt
NoOpNoOp$^lstm_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2J
#lstm_cell_1/StatefulPartitionedCall#lstm_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Р

(__inference_dense_2_layer_call_fn_306833

inputs
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_302644o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
A
Ё

lstm_2_while_body_304231*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3)
%lstm_2_while_lstm_2_strided_slice_1_0e
alstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0K
9lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0:(M
;lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0:
(H
:lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0:(
lstm_2_while_identity
lstm_2_while_identity_1
lstm_2_while_identity_2
lstm_2_while_identity_3
lstm_2_while_identity_4
lstm_2_while_identity_5'
#lstm_2_while_lstm_2_strided_slice_1c
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensorI
7lstm_2_while_lstm_cell_2_matmul_readvariableop_resource:(K
9lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource:
(F
8lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource:(Ђ/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOpЂ.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOpЂ0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Щ
0lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0lstm_2_while_placeholderGlstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ј
.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp9lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:(*
dtype0Ь
lstm_2/while/lstm_cell_2/MatMulMatMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(Ќ
0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp;lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype0Г
!lstm_2/while/lstm_cell_2/MatMul_1MatMullstm_2_while_placeholder_28lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(Џ
lstm_2/while/lstm_cell_2/addAddV2)lstm_2/while/lstm_cell_2/MatMul:product:0+lstm_2/while/lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(І
/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp:lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype0И
 lstm_2/while/lstm_cell_2/BiasAddBiasAdd lstm_2/while/lstm_cell_2/add:z:07lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(j
(lstm_2/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_2/while/lstm_cell_2/splitSplit1lstm_2/while/lstm_cell_2/split/split_dim:output:0)lstm_2/while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_split
 lstm_2/while/lstm_cell_2/SigmoidSigmoid'lstm_2/while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ

"lstm_2/while/lstm_cell_2/Sigmoid_1Sigmoid'lstm_2/while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_2/while/lstm_cell_2/mulMul&lstm_2/while/lstm_cell_2/Sigmoid_1:y:0lstm_2_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_2/while/lstm_cell_2/ReluRelu'lstm_2/while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
lstm_2/while/lstm_cell_2/mul_1Mul$lstm_2/while/lstm_cell_2/Sigmoid:y:0+lstm_2/while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_2/while/lstm_cell_2/add_1AddV2 lstm_2/while/lstm_cell_2/mul:z:0"lstm_2/while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

"lstm_2/while/lstm_cell_2/Sigmoid_2Sigmoid'lstm_2/while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
}
lstm_2/while/lstm_cell_2/Relu_1Relu"lstm_2/while/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ў
lstm_2/while/lstm_cell_2/mul_2Mul&lstm_2/while/lstm_cell_2/Sigmoid_2:y:0-lstm_2/while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
y
7lstm_2/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
1lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_2_while_placeholder_1@lstm_2/while/TensorArrayV2Write/TensorListSetItem/index:output:0"lstm_2/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвT
lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_2/while/addAddV2lstm_2_while_placeholderlstm_2/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_2/while/add_1AddV2&lstm_2_while_lstm_2_while_loop_counterlstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_2/while/IdentityIdentitylstm_2/while/add_1:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: 
lstm_2/while/Identity_1Identity,lstm_2_while_lstm_2_while_maximum_iterations^lstm_2/while/NoOp*
T0*
_output_shapes
: n
lstm_2/while/Identity_2Identitylstm_2/while/add:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: 
lstm_2/while/Identity_3IdentityAlstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_2/while/NoOp*
T0*
_output_shapes
: 
lstm_2/while/Identity_4Identity"lstm_2/while/lstm_cell_2/mul_2:z:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_2/while/Identity_5Identity"lstm_2/while/lstm_cell_2/add_1:z:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
щ
lstm_2/while/NoOpNoOp0^lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_2_while_identitylstm_2/while/Identity:output:0";
lstm_2_while_identity_1 lstm_2/while/Identity_1:output:0";
lstm_2_while_identity_2 lstm_2/while/Identity_2:output:0";
lstm_2_while_identity_3 lstm_2/while/Identity_3:output:0";
lstm_2_while_identity_4 lstm_2/while/Identity_4:output:0";
lstm_2_while_identity_5 lstm_2/while/Identity_5:output:0"L
#lstm_2_while_lstm_2_strided_slice_1%lstm_2_while_lstm_2_strided_slice_1_0"v
8lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource:lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0"x
9lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource;lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0"t
7lstm_2_while_lstm_cell_2_matmul_readvariableop_resource9lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0"Ф
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensoralstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : 2b
/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp2`
.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp2d
0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
Ъ

G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_301445

inputs

states
states_10
matmul_readvariableop_resource:P2
 matmul_1_readvariableop_resource:P-
biasadd_readvariableop_resource:P
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџPr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџN
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates
Л7
С
while_body_306036
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_1_matmul_readvariableop_resource_0:PF
4while_lstm_cell_1_matmul_1_readvariableop_resource_0:PA
3while_lstm_cell_1_biasadd_readvariableop_resource_0:P
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_1_matmul_readvariableop_resource:PD
2while_lstm_cell_1_matmul_1_readvariableop_resource:P?
1while_lstm_cell_1_biasadd_readvariableop_resource:PЂ(while/lstm_cell_1/BiasAdd/ReadVariableOpЂ'while/lstm_cell_1/MatMul/ReadVariableOpЂ)while/lstm_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
'while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0З
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
)while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype0
while/lstm_cell_1/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
while/lstm_cell_1/addAddV2"while/lstm_cell_1/MatMul:product:0$while/lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP
(while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0Ѓ
while/lstm_cell_1/BiasAddBiasAddwhile/lstm_cell_1/add:z:00while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPc
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ь
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0"while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitx
while/lstm_cell_1/SigmoidSigmoid while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџz
while/lstm_cell_1/Sigmoid_1Sigmoid while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mulMulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџr
while/lstm_cell_1/ReluRelu while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mul_1Mulwhile/lstm_cell_1/Sigmoid:y:0$while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/add_1AddV2while/lstm_cell_1/mul:z:0while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџz
while/lstm_cell_1/Sigmoid_2Sigmoid while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџo
while/lstm_cell_1/Relu_1Reluwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mul_2Mulwhile/lstm_cell_1/Sigmoid_2:y:0&while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_1/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџx
while/Identity_5Identitywhile/lstm_cell_1/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџЭ

while/NoOpNoOp)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_1_biasadd_readvariableop_resource3while_lstm_cell_1_biasadd_readvariableop_resource_0"j
2while_lstm_cell_1_matmul_1_readvariableop_resource4while_lstm_cell_1_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_1_matmul_readvariableop_resource2while_lstm_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2T
(while/lstm_cell_1/BiasAdd/ReadVariableOp(while/lstm_cell_1/BiasAdd/ReadVariableOp2R
'while/lstm_cell_1/MatMul/ReadVariableOp'while/lstm_cell_1/MatMul/ReadVariableOp2V
)while/lstm_cell_1/MatMul_1/ReadVariableOp)while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Ѕ<
Я	
lstm_while_body_304432&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0G
5lstm_while_lstm_cell_matmul_readvariableop_resource_0:ExI
7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:xD
6lstm_while_lstm_cell_biasadd_readvariableop_resource_0:x
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorE
3lstm_while_lstm_cell_matmul_readvariableop_resource:ExG
5lstm_while_lstm_cell_matmul_1_readvariableop_resource:xB
4lstm_while_lstm_cell_biasadd_readvariableop_resource:xЂ+lstm/while/lstm_cell/BiasAdd/ReadVariableOpЂ*lstm/while/lstm_cell/MatMul/ReadVariableOpЂ,lstm/while/lstm_cell/MatMul_1/ReadVariableOp
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   П
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџE*
element_dtype0 
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:Ex*
dtype0Т
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxЄ
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0Љ
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxЃ
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџx
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0Ќ
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxf
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ѕ
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split~
lstm/while/lstm_cell/SigmoidSigmoid#lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/while/lstm_cell/Sigmoid_1Sigmoid#lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/while/lstm_cell/mulMul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџx
lstm/while/lstm_cell/ReluRelu#lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/while/lstm_cell/mul_1Mul lstm/while/lstm_cell/Sigmoid:y:0'lstm/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/while/lstm_cell/Sigmoid_2Sigmoid#lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџu
lstm/while/lstm_cell/Relu_1Relulstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
lstm/while/lstm_cell/mul_2Mul"lstm/while/lstm_cell/Sigmoid_2:y:0)lstm/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџж
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвR
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: T
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: h
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ~
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: h
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџл
lstm/while/NoOpNoOp,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"n
4lstm_while_lstm_cell_biasadd_readvariableop_resource6lstm_while_lstm_cell_biasadd_readvariableop_resource_0"p
5lstm_while_lstm_cell_matmul_1_readvariableop_resource7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"l
3lstm_while_lstm_cell_matmul_readvariableop_resource5lstm_while_lstm_cell_matmul_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"М
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2Z
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp+lstm/while/lstm_cell/BiasAdd/ReadVariableOp2X
*lstm/while/lstm_cell/MatMul/ReadVariableOp*lstm/while/lstm_cell/MatMul/ReadVariableOp2\
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp,lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
в

G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_307068

inputs
states_0
states_10
matmul_readvariableop_resource:P2
 matmul_1_readvariableop_resource:P-
biasadd_readvariableop_resource:P
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџPr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџN
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/1
а

E__inference_lstm_cell_layer_call_and_return_conditional_losses_306970

inputs
states_0
states_10
matmul_readvariableop_resource:Ex2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Ex*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџN
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџE:џџџџџџџџџ:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/1
Г6
Џ
while_body_305250
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:ExD
2while_lstm_cell_matmul_1_readvariableop_resource_0:x?
1while_lstm_cell_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:ExB
0while_lstm_cell_matmul_1_readvariableop_resource:x=
/while_lstm_cell_biasadd_readvariableop_resource:xЂ&while/lstm_cell/BiasAdd/ReadVariableOpЂ%while/lstm_cell/MatMul/ReadVariableOpЂ'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџE*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:Ex*
dtype0Г
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџx
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџv
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџn
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџv
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџk
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџТ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџv
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџЧ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
ц
ђ
,__inference_lstm_cell_2_layer_call_fn_307117

inputs
states_0
states_1
unknown:(
	unknown_0:
(
	unknown_1:(
identity

identity_1

identity_2ЂStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_301795o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
states/1
ђ
Б
'__inference_lstm_2_layer_call_fn_306164

inputs
unknown:(
	unknown_0:
(
	unknown_1:(
identityЂStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_302979o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Л7
С
while_body_305750
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_1_matmul_readvariableop_resource_0:PF
4while_lstm_cell_1_matmul_1_readvariableop_resource_0:PA
3while_lstm_cell_1_biasadd_readvariableop_resource_0:P
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_1_matmul_readvariableop_resource:PD
2while_lstm_cell_1_matmul_1_readvariableop_resource:P?
1while_lstm_cell_1_biasadd_readvariableop_resource:PЂ(while/lstm_cell_1/BiasAdd/ReadVariableOpЂ'while/lstm_cell_1/MatMul/ReadVariableOpЂ)while/lstm_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
'while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0З
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
)while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype0
while/lstm_cell_1/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
while/lstm_cell_1/addAddV2"while/lstm_cell_1/MatMul:product:0$while/lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP
(while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0Ѓ
while/lstm_cell_1/BiasAddBiasAddwhile/lstm_cell_1/add:z:00while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџPc
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ь
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0"while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitx
while/lstm_cell_1/SigmoidSigmoid while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџz
while/lstm_cell_1/Sigmoid_1Sigmoid while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mulMulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџr
while/lstm_cell_1/ReluRelu while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mul_1Mulwhile/lstm_cell_1/Sigmoid:y:0$while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/add_1AddV2while/lstm_cell_1/mul:z:0while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџz
while/lstm_cell_1/Sigmoid_2Sigmoid while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџo
while/lstm_cell_1/Relu_1Reluwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell_1/mul_2Mulwhile/lstm_cell_1/Sigmoid_2:y:0&while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџФ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_1/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџx
while/Identity_5Identitywhile/lstm_cell_1/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџЭ

while/NoOpNoOp)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_1_biasadd_readvariableop_resource3while_lstm_cell_1_biasadd_readvariableop_resource_0"j
2while_lstm_cell_1_matmul_1_readvariableop_resource4while_lstm_cell_1_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_1_matmul_readvariableop_resource2while_lstm_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2T
(while/lstm_cell_1/BiasAdd/ReadVariableOp(while/lstm_cell_1/BiasAdd/ReadVariableOp2R
'while/lstm_cell_1/MatMul/ReadVariableOp'while/lstm_cell_1/MatMul/ReadVariableOp2V
)while/lstm_cell_1/MatMul_1/ReadVariableOp)while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Е
У
while_cond_302002
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_302002___redundant_placeholder04
0while_while_cond_302002___redundant_placeholder14
0while_while_cond_302002___redundant_placeholder24
0while_while_cond_302002___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ
:џџџџџџџџџ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
ф#
ж
while_body_301810
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_2_301834_0:(,
while_lstm_cell_2_301836_0:
((
while_lstm_cell_2_301838_0:(
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_2_301834:(*
while_lstm_cell_2_301836:
(&
while_lstm_cell_2_301838:(Ђ)while/lstm_cell_2/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ў
)while/lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_2_301834_0while_lstm_cell_2_301836_0while_lstm_cell_2_301838_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_301795r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:02while/lstm_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity2while/lstm_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

while/Identity_5Identity2while/lstm_cell_2/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
x

while/NoOpNoOp*^while/lstm_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_2_301834while_lstm_cell_2_301834_0"6
while_lstm_cell_2_301836while_lstm_cell_2_301836_0"6
while_lstm_cell_2_301838while_lstm_cell_2_301838_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : 2V
)while/lstm_cell_2/StatefulPartitionedCall)while/lstm_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
Ш"
ж
while_body_301650
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_1_301674_0:P,
while_lstm_cell_1_301676_0:P(
while_lstm_cell_1_301678_0:P
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_1_301674:P*
while_lstm_cell_1_301676:P&
while_lstm_cell_1_301678:PЂ)while/lstm_cell_1/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ў
)while/lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1_301674_0while_lstm_cell_1_301676_0while_lstm_cell_1_301678_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_301591л
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity2while/lstm_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
while/Identity_5Identity2while/lstm_cell_1/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџx

while/NoOpNoOp*^while/lstm_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_1_301674while_lstm_cell_1_301674_0"6
while_lstm_cell_1_301676while_lstm_cell_1_301676_0"6
while_lstm_cell_1_301678while_lstm_cell_1_301678_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2V
)while/lstm_cell_1/StatefulPartitionedCall)while/lstm_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
ЂI

@__inference_lstm_layer_call_and_return_conditional_losses_305048
inputs_0:
(lstm_cell_matmul_readvariableop_resource:Ex<
*lstm_cell_matmul_1_readvariableop_resource:x7
)lstm_cell_biasadd_readvariableop_resource:x
identityЂ lstm_cell/BiasAdd/ReadVariableOpЂlstm_cell/MatMul/ReadVariableOpЂ!lstm_cell/MatMul_1/ReadVariableOpЂwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџED
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџE*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

:Ex*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџx
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџq
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџr
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџj
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : љ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_304964*
condR
while_cond_304963*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџЗ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџE: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџE
"
_user_specified_name
inputs/0
нI

B__inference_lstm_1_layer_call_and_return_conditional_losses_306120

inputs<
*lstm_cell_1_matmul_readvariableop_resource:P>
,lstm_cell_1_matmul_1_readvariableop_resource:P9
+lstm_cell_1_biasadd_readvariableop_resource:P
identityЂ"lstm_cell_1/BiasAdd/ReadVariableOpЂ!lstm_cell_1/MatMul/ReadVariableOpЂ#lstm_cell_1/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype0
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :к
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitl
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџn
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџu
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
lstm_cell_1/ReluRelulstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџx
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџc
lstm_cell_1/Relu_1Relulstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0 lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_1_matmul_readvariableop_resource,lstm_cell_1_matmul_1_readvariableop_resource+lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_306036*
condR
while_cond_306035*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Т
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџН
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ш

,__inference_fingerprint_layer_call_fn_306773

inputs
unknown:
@
	unknown_0:@
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_fingerprint_layer_call_and_return_conditional_losses_302576o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
Е
У
while_cond_306368
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_306368___redundant_placeholder04
0while_while_cond_306368___redundant_placeholder14
0while_while_cond_306368___redundant_placeholder24
0while_while_cond_306368___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ
:џџџџџџџџџ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
џ7
§
B__inference_lstm_1_layer_call_and_return_conditional_losses_301719

inputs$
lstm_cell_1_301637:P$
lstm_cell_1_301639:P 
lstm_cell_1_301641:P
identityЂ#lstm_cell_1/StatefulPartitionedCallЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask№
#lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1_301637lstm_cell_1_301639lstm_cell_1_301641*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_301591n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Д
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1_301637lstm_cell_1_301639lstm_cell_1_301641*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_301650*
condR
while_cond_301649*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџt
NoOpNoOp$^lstm_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2J
#lstm_cell_1/StatefulPartitionedCall#lstm_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
в

G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_307198

inputs
states_0
states_10
matmul_readvariableop_resource:(2
 matmul_1_readvariableop_resource:
(-
biasadd_readvariableop_resource:(
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ
N
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
states/1
Е
У
while_cond_302454
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_302454___redundant_placeholder04
0while_while_cond_302454___redundant_placeholder14
0while_while_cond_302454___redundant_placeholder24
0while_while_cond_302454___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ
:џџџџџџџџџ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
Е
У
while_cond_306223
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_306223___redundant_placeholder04
0while_while_cond_306223___redundant_placeholder14
0while_while_cond_306223___redundant_placeholder24
0while_while_cond_306223___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ
:џџџџџџџџџ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
Е
У
while_cond_303059
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_303059___redundant_placeholder04
0while_while_cond_303059___redundant_placeholder14
0while_while_cond_303059___redundant_placeholder24
0while_while_cond_303059___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Е
У
while_cond_303247
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_303247___redundant_placeholder04
0while_while_cond_303247___redundant_placeholder14
0while_while_cond_303247___redundant_placeholder24
0while_while_cond_303247___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Ђ
Ч
model_lstm_1_while_cond_3007456
2model_lstm_1_while_model_lstm_1_while_loop_counter<
8model_lstm_1_while_model_lstm_1_while_maximum_iterations"
model_lstm_1_while_placeholder$
 model_lstm_1_while_placeholder_1$
 model_lstm_1_while_placeholder_2$
 model_lstm_1_while_placeholder_38
4model_lstm_1_while_less_model_lstm_1_strided_slice_1N
Jmodel_lstm_1_while_model_lstm_1_while_cond_300745___redundant_placeholder0N
Jmodel_lstm_1_while_model_lstm_1_while_cond_300745___redundant_placeholder1N
Jmodel_lstm_1_while_model_lstm_1_while_cond_300745___redundant_placeholder2N
Jmodel_lstm_1_while_model_lstm_1_while_cond_300745___redundant_placeholder3
model_lstm_1_while_identity

model/lstm_1/while/LessLessmodel_lstm_1_while_placeholder4model_lstm_1_while_less_model_lstm_1_strided_slice_1*
T0*
_output_shapes
: e
model/lstm_1/while/IdentityIdentitymodel/lstm_1/while/Less:z:0*
T0
*
_output_shapes
: "C
model_lstm_1_while_identity$model/lstm_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:


є
C__inference_dense_2_layer_call_and_return_conditional_losses_302644

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Є

љ
H__inference_class_output_layer_call_and_return_conditional_losses_302678

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
A
Ё

lstm_2_while_body_304719*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3)
%lstm_2_while_lstm_2_strided_slice_1_0e
alstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0K
9lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0:(M
;lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0:
(H
:lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0:(
lstm_2_while_identity
lstm_2_while_identity_1
lstm_2_while_identity_2
lstm_2_while_identity_3
lstm_2_while_identity_4
lstm_2_while_identity_5'
#lstm_2_while_lstm_2_strided_slice_1c
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensorI
7lstm_2_while_lstm_cell_2_matmul_readvariableop_resource:(K
9lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource:
(F
8lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource:(Ђ/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOpЂ.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOpЂ0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Щ
0lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0lstm_2_while_placeholderGlstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ј
.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp9lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:(*
dtype0Ь
lstm_2/while/lstm_cell_2/MatMulMatMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(Ќ
0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp;lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype0Г
!lstm_2/while/lstm_cell_2/MatMul_1MatMullstm_2_while_placeholder_28lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(Џ
lstm_2/while/lstm_cell_2/addAddV2)lstm_2/while/lstm_cell_2/MatMul:product:0+lstm_2/while/lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(І
/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp:lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype0И
 lstm_2/while/lstm_cell_2/BiasAddBiasAdd lstm_2/while/lstm_cell_2/add:z:07lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(j
(lstm_2/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_2/while/lstm_cell_2/splitSplit1lstm_2/while/lstm_cell_2/split/split_dim:output:0)lstm_2/while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_split
 lstm_2/while/lstm_cell_2/SigmoidSigmoid'lstm_2/while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ

"lstm_2/while/lstm_cell_2/Sigmoid_1Sigmoid'lstm_2/while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_2/while/lstm_cell_2/mulMul&lstm_2/while/lstm_cell_2/Sigmoid_1:y:0lstm_2_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_2/while/lstm_cell_2/ReluRelu'lstm_2/while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
lstm_2/while/lstm_cell_2/mul_1Mul$lstm_2/while/lstm_cell_2/Sigmoid:y:0+lstm_2/while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_2/while/lstm_cell_2/add_1AddV2 lstm_2/while/lstm_cell_2/mul:z:0"lstm_2/while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

"lstm_2/while/lstm_cell_2/Sigmoid_2Sigmoid'lstm_2/while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
}
lstm_2/while/lstm_cell_2/Relu_1Relu"lstm_2/while/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ў
lstm_2/while/lstm_cell_2/mul_2Mul&lstm_2/while/lstm_cell_2/Sigmoid_2:y:0-lstm_2/while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
y
7lstm_2/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
1lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_2_while_placeholder_1@lstm_2/while/TensorArrayV2Write/TensorListSetItem/index:output:0"lstm_2/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвT
lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_2/while/addAddV2lstm_2_while_placeholderlstm_2/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_2/while/add_1AddV2&lstm_2_while_lstm_2_while_loop_counterlstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_2/while/IdentityIdentitylstm_2/while/add_1:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: 
lstm_2/while/Identity_1Identity,lstm_2_while_lstm_2_while_maximum_iterations^lstm_2/while/NoOp*
T0*
_output_shapes
: n
lstm_2/while/Identity_2Identitylstm_2/while/add:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: 
lstm_2/while/Identity_3IdentityAlstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_2/while/NoOp*
T0*
_output_shapes
: 
lstm_2/while/Identity_4Identity"lstm_2/while/lstm_cell_2/mul_2:z:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_2/while/Identity_5Identity"lstm_2/while/lstm_cell_2/add_1:z:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
щ
lstm_2/while/NoOpNoOp0^lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_2_while_identitylstm_2/while/Identity:output:0";
lstm_2_while_identity_1 lstm_2/while/Identity_1:output:0";
lstm_2_while_identity_2 lstm_2/while/Identity_2:output:0";
lstm_2_while_identity_3 lstm_2/while/Identity_3:output:0";
lstm_2_while_identity_4 lstm_2/while/Identity_4:output:0";
lstm_2_while_identity_5 lstm_2/while/Identity_5:output:0"L
#lstm_2_while_lstm_2_strided_slice_1%lstm_2_while_lstm_2_strided_slice_1_0"v
8lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource:lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0"x
9lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource;lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0"t
7lstm_2_while_lstm_cell_2_matmul_readvariableop_resource9lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0"Ф
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensoralstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : 2b
/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp2`
.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp2d
0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
п7
ѓ
@__inference_lstm_layer_call_and_return_conditional_losses_301178

inputs"
lstm_cell_301096:Ex"
lstm_cell_301098:x
lstm_cell_301100:x
identityЂ!lstm_cell/StatefulPartitionedCallЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџED
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџE*
shrink_axis_maskц
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_301096lstm_cell_301098lstm_cell_301100*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_301095n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ў
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_301096lstm_cell_301098lstm_cell_301100*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_301109*
condR
while_cond_301108*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџr
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџE: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџE
 
_user_specified_nameinputs
Е
У
while_cond_306035
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_306035___redundant_placeholder04
0while_while_cond_306035___redundant_placeholder14
0while_while_cond_306035___redundant_placeholder24
0while_while_cond_306035___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
нI

B__inference_lstm_1_layer_call_and_return_conditional_losses_302388

inputs<
*lstm_cell_1_matmul_readvariableop_resource:P>
,lstm_cell_1_matmul_1_readvariableop_resource:P9
+lstm_cell_1_biasadd_readvariableop_resource:P
identityЂ"lstm_cell_1/BiasAdd/ReadVariableOpЂ!lstm_cell_1/MatMul/ReadVariableOpЂ#lstm_cell_1/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype0
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :к
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitl
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџn
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџu
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
lstm_cell_1/ReluRelulstm_cell_1/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџx
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџc
lstm_cell_1/Relu_1Relulstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0 lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_1_matmul_readvariableop_resource,lstm_cell_1_matmul_1_readvariableop_resource+lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_302304*
condR
while_cond_302303*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Т
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџН
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
а

E__inference_lstm_cell_layer_call_and_return_conditional_losses_307002

inputs
states_0
states_10
matmul_readvariableop_resource:Ex2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Ex*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџN
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџE:џџџџџџџџџ:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџE
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/1
з8
С
while_body_306224
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_2_matmul_readvariableop_resource_0:(F
4while_lstm_cell_2_matmul_1_readvariableop_resource_0:
(A
3while_lstm_cell_2_biasadd_readvariableop_resource_0:(
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_2_matmul_readvariableop_resource:(D
2while_lstm_cell_2_matmul_1_readvariableop_resource:
(?
1while_lstm_cell_2_biasadd_readvariableop_resource:(Ђ(while/lstm_cell_2/BiasAdd/ReadVariableOpЂ'while/lstm_cell_2/MatMul/ReadVariableOpЂ)while/lstm_cell_2/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:(*
dtype0З
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype0
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ(
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype0Ѓ
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(c
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ь
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
:џџџџџџџџџ
*
	num_splitx
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
z
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ
r
while/lstm_cell_2/ReluRelu while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0$while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
z
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ
o
while/lstm_cell_2/Relu_1Reluwhile/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0&while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
x
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Э

while/NoOpNoOp)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ
:џџџџџџџџџ
: : : : : 2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ
:-)
'
_output_shapes
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
Г6
Џ
while_body_303248
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0:ExD
2while_lstm_cell_matmul_1_readvariableop_resource_0:x?
1while_lstm_cell_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource:ExB
0while_lstm_cell_matmul_1_readvariableop_resource:x=
/while_lstm_cell_biasadd_readvariableop_resource:xЂ&while/lstm_cell/BiasAdd/ReadVariableOpЂ%while/lstm_cell/MatMul/ReadVariableOpЂ'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџE   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџE*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

:Ex*
dtype0Г
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџx
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџv
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџn
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџv
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџk
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџТ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџv
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџЧ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Е
У
while_cond_305106
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_305106___redundant_placeholder04
0while_while_cond_305106___redundant_placeholder14
0while_while_cond_305106___redundant_placeholder24
0while_while_cond_305106___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:

Г
'__inference_lstm_2_layer_call_fn_306142
inputs_0
unknown:(
	unknown_0:
(
	unknown_1:(
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_302073o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0"Е	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ј
serving_defaultф
?
input_14
serving_default_input_1:0џџџџџџџџџE@
class_output0
StatefulPartitionedCall:0џџџџџџџџџC
identity_output0
StatefulPartitionedCall:1џџџџџџџџџtensorflow/serving/predict:Ќх
И
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures"
_tf_keras_network
"
_tf_keras_input_layer
к
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell
 
state_spec"
_tf_keras_rnn_layer
М
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator"
_tf_keras_layer
к
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
._random_generator
/cell
0
state_spec"
_tf_keras_rnn_layer
к
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7_random_generator
8cell
9
state_spec"
_tf_keras_rnn_layer
Л
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias"
_tf_keras_layer
Л
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias"
_tf_keras_layer
Л
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

Pkernel
Qbias"
_tf_keras_layer
Л
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses

Xkernel
Ybias"
_tf_keras_layer
Л
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias"
_tf_keras_layer
Л
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias"
_tf_keras_layer
Л
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses

pkernel
qbias"
_tf_keras_layer
Л
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

xkernel
ybias"
_tf_keras_layer
с
z0
{1
|2
}3
~4
5
6
7
8
@9
A10
H11
I12
P13
Q14
X15
Y16
`17
a18
h19
i20
p21
q22
x23
y24"
trackable_list_wrapper
с
z0
{1
|2
}3
~4
5
6
7
8
@9
A10
H11
I12
P13
Q14
X15
Y16
`17
a18
h19
i20
p21
q22
x23
y24"
trackable_list_wrapper
 "
trackable_list_wrapper
Я
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
е
trace_0
trace_1
trace_2
trace_32т
&__inference_model_layer_call_fn_302741
&__inference_model_layer_call_fn_303835
&__inference_model_layer_call_fn_303892
&__inference_model_layer_call_fn_303579П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
С
trace_0
trace_1
trace_2
trace_32Ю
A__inference_model_layer_call_and_return_conditional_losses_304373
A__inference_model_layer_call_and_return_conditional_losses_304861
A__inference_model_layer_call_and_return_conditional_losses_303646
A__inference_model_layer_call_and_return_conditional_losses_303713П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
ЬBЩ
!__inference__wrapped_model_301028input_1"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в
	iter
beta_1
beta_2

decay
learning_rate@mЯAmаHmбImвPmгQmдXmеYmж`mзamиhmйimкpmлqmмxmнymоzmп{mр|mс}mт~mуmф	mх	mц	mч@vшAvщHvъIvыPvьQvэXvюYvя`v№avёhvђivѓpvєqvѕxvіyvїzvј{vљ|vњ}vћ~vќv§	vў	vџ	v"
	optimizer
 "
trackable_dict_wrapper
-
serving_default"
signature_map
5
z0
{1
|2"
trackable_list_wrapper
5
z0
{1
|2"
trackable_list_wrapper
 "
trackable_list_wrapper
П
states
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ц
trace_0
trace_1
trace_2
trace_32ѓ
%__inference_lstm_layer_call_fn_304872
%__inference_lstm_layer_call_fn_304883
%__inference_lstm_layer_call_fn_304894
%__inference_lstm_layer_call_fn_304905д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
в
 trace_0
Ёtrace_1
Ђtrace_2
Ѓtrace_32п
@__inference_lstm_layer_call_and_return_conditional_losses_305048
@__inference_lstm_layer_call_and_return_conditional_losses_305191
@__inference_lstm_layer_call_and_return_conditional_losses_305334
@__inference_lstm_layer_call_and_return_conditional_losses_305477д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z trace_0zЁtrace_1zЂtrace_2zЃtrace_3
"
_generic_user_object

Є	variables
Ѕtrainable_variables
Іregularization_losses
Ї	keras_api
Ј__call__
+Љ&call_and_return_all_conditional_losses
Њ_random_generator
Ћ
state_size

zkernel
{recurrent_kernel
|bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ќnon_trainable_variables
­layers
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
Х
Бtrace_0
Вtrace_12
(__inference_dropout_layer_call_fn_305482
(__inference_dropout_layer_call_fn_305487Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zБtrace_0zВtrace_1
ћ
Гtrace_0
Дtrace_12Р
C__inference_dropout_layer_call_and_return_conditional_losses_305492
C__inference_dropout_layer_call_and_return_conditional_losses_305504Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zГtrace_0zДtrace_1
"
_generic_user_object
5
}0
~1
2"
trackable_list_wrapper
5
}0
~1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
П
Еstates
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
ю
Лtrace_0
Мtrace_1
Нtrace_2
Оtrace_32ћ
'__inference_lstm_1_layer_call_fn_305515
'__inference_lstm_1_layer_call_fn_305526
'__inference_lstm_1_layer_call_fn_305537
'__inference_lstm_1_layer_call_fn_305548д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЛtrace_0zМtrace_1zНtrace_2zОtrace_3
к
Пtrace_0
Рtrace_1
Сtrace_2
Тtrace_32ч
B__inference_lstm_1_layer_call_and_return_conditional_losses_305691
B__inference_lstm_1_layer_call_and_return_conditional_losses_305834
B__inference_lstm_1_layer_call_and_return_conditional_losses_305977
B__inference_lstm_1_layer_call_and_return_conditional_losses_306120д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zПtrace_0zРtrace_1zСtrace_2zТtrace_3
"
_generic_user_object

У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses
Щ_random_generator
Ъ
state_size

}kernel
~recurrent_kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
П
Ыstates
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
ю
бtrace_0
вtrace_1
гtrace_2
дtrace_32ћ
'__inference_lstm_2_layer_call_fn_306131
'__inference_lstm_2_layer_call_fn_306142
'__inference_lstm_2_layer_call_fn_306153
'__inference_lstm_2_layer_call_fn_306164д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zбtrace_0zвtrace_1zгtrace_2zдtrace_3
к
еtrace_0
жtrace_1
зtrace_2
иtrace_32ч
B__inference_lstm_2_layer_call_and_return_conditional_losses_306309
B__inference_lstm_2_layer_call_and_return_conditional_losses_306454
B__inference_lstm_2_layer_call_and_return_conditional_losses_306599
B__inference_lstm_2_layer_call_and_return_conditional_losses_306744д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zеtrace_0zжtrace_1zзtrace_2zиtrace_3
"
_generic_user_object

й	variables
кtrainable_variables
лregularization_losses
м	keras_api
н__call__
+о&call_and_return_all_conditional_losses
п_random_generator
р
state_size
kernel
recurrent_kernel
	bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
ь
цtrace_02Э
&__inference_dense_layer_call_fn_306753Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zцtrace_0

чtrace_02ш
A__inference_dense_layer_call_and_return_conditional_losses_306764Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zчtrace_0
:

2dense/kernel
:
2
dense/bias
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
ђ
эtrace_02г
,__inference_fingerprint_layer_call_fn_306773Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zэtrace_0

юtrace_02ю
G__inference_fingerprint_layer_call_and_return_conditional_losses_306784Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zюtrace_0
$:"
@2fingerprint/kernel
:@2fingerprint/bias
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
яnon_trainable_variables
№layers
ёmetrics
 ђlayer_regularization_losses
ѓlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
ю
єtrace_02Я
(__inference_dense_1_layer_call_fn_306793Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zєtrace_0

ѕtrace_02ъ
C__inference_dense_1_layer_call_and_return_conditional_losses_306804Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zѕtrace_0
 :@ 2dense_1/kernel
: 2dense_1/bias
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
іnon_trainable_variables
їlayers
јmetrics
 љlayer_regularization_losses
њlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
ю
ћtrace_02Я
(__inference_dense_3_layer_call_fn_306813Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zћtrace_0

ќtrace_02ъ
C__inference_dense_3_layer_call_and_return_conditional_losses_306824Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zќtrace_0
 :@ 2dense_3/kernel
: 2dense_3/bias
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
§non_trainable_variables
ўlayers
џmetrics
 layer_regularization_losses
layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
ю
trace_02Я
(__inference_dense_2_layer_call_fn_306833Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ъ
C__inference_dense_2_layer_call_and_return_conditional_losses_306844Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 : 2dense_2/kernel
:2dense_2/bias
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
ю
trace_02Я
(__inference_dense_4_layer_call_fn_306853Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ъ
C__inference_dense_4_layer_call_and_return_conditional_losses_306864Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 : 2dense_4/kernel
:2dense_4/bias
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
ѓ
trace_02д
-__inference_class_output_layer_call_fn_306873Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02я
H__inference_class_output_layer_call_and_return_conditional_losses_306884Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
%:#2class_output/kernel
:2class_output/bias
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
і
trace_02з
0__inference_identity_output_layer_call_fn_306893Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ђ
K__inference_identity_output_layer_call_and_return_conditional_losses_306904Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
(:&2identity_output/kernel
": 2identity_output/bias
':%Ex2lstm/lstm_cell/kernel
1:/x2lstm/lstm_cell/recurrent_kernel
!:x2lstm/lstm_cell/bias
+:)P2lstm_1/lstm_cell_1/kernel
5:3P2#lstm_1/lstm_cell_1/recurrent_kernel
%:#P2lstm_1/lstm_cell_1/bias
+:)(2lstm_2/lstm_cell_2/kernel
5:3
(2#lstm_2/lstm_cell_2/recurrent_kernel
%:#(2lstm_2/lstm_cell_2/bias
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
H
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
јBѕ
&__inference_model_layer_call_fn_302741input_1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
&__inference_model_layer_call_fn_303835inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
&__inference_model_layer_call_fn_303892inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
&__inference_model_layer_call_fn_303579input_1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
A__inference_model_layer_call_and_return_conditional_losses_304373inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
A__inference_model_layer_call_and_return_conditional_losses_304861inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
A__inference_model_layer_call_and_return_conditional_losses_303646input_1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
A__inference_model_layer_call_and_return_conditional_losses_303713input_1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ЫBШ
$__inference_signature_wrapper_303778input_1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
%__inference_lstm_layer_call_fn_304872inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
%__inference_lstm_layer_call_fn_304883inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
%__inference_lstm_layer_call_fn_304894inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
%__inference_lstm_layer_call_fn_304905inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЈBЅ
@__inference_lstm_layer_call_and_return_conditional_losses_305048inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЈBЅ
@__inference_lstm_layer_call_and_return_conditional_losses_305191inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ІBЃ
@__inference_lstm_layer_call_and_return_conditional_losses_305334inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ІBЃ
@__inference_lstm_layer_call_and_return_conditional_losses_305477inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
5
z0
{1
|2"
trackable_list_wrapper
5
z0
{1
|2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
Є	variables
Ѕtrainable_variables
Іregularization_losses
Ј__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
г
Ѓtrace_0
Єtrace_12
*__inference_lstm_cell_layer_call_fn_306921
*__inference_lstm_cell_layer_call_fn_306938Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЃtrace_0zЄtrace_1

Ѕtrace_0
Іtrace_12Ю
E__inference_lstm_cell_layer_call_and_return_conditional_losses_306970
E__inference_lstm_cell_layer_call_and_return_conditional_losses_307002Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЅtrace_0zІtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBъ
(__inference_dropout_layer_call_fn_305482inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
(__inference_dropout_layer_call_fn_305487inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_dropout_layer_call_and_return_conditional_losses_305492inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_dropout_layer_call_and_return_conditional_losses_305504inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
'__inference_lstm_1_layer_call_fn_305515inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
'__inference_lstm_1_layer_call_fn_305526inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
'__inference_lstm_1_layer_call_fn_305537inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
'__inference_lstm_1_layer_call_fn_305548inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЊBЇ
B__inference_lstm_1_layer_call_and_return_conditional_losses_305691inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЊBЇ
B__inference_lstm_1_layer_call_and_return_conditional_losses_305834inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЈBЅ
B__inference_lstm_1_layer_call_and_return_conditional_losses_305977inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЈBЅ
B__inference_lstm_1_layer_call_and_return_conditional_losses_306120inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
5
}0
~1
2"
trackable_list_wrapper
5
}0
~1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
з
Ќtrace_0
­trace_12
,__inference_lstm_cell_1_layer_call_fn_307019
,__inference_lstm_cell_1_layer_call_fn_307036Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЌtrace_0z­trace_1

Ўtrace_0
Џtrace_12в
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_307068
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_307100Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЎtrace_0zЏtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
80"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
'__inference_lstm_2_layer_call_fn_306131inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
'__inference_lstm_2_layer_call_fn_306142inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
'__inference_lstm_2_layer_call_fn_306153inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
'__inference_lstm_2_layer_call_fn_306164inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЊBЇ
B__inference_lstm_2_layer_call_and_return_conditional_losses_306309inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЊBЇ
B__inference_lstm_2_layer_call_and_return_conditional_losses_306454inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЈBЅ
B__inference_lstm_2_layer_call_and_return_conditional_losses_306599inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЈBЅ
B__inference_lstm_2_layer_call_and_return_conditional_losses_306744inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
8
0
1
2"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
й	variables
кtrainable_variables
лregularization_losses
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
з
Еtrace_0
Жtrace_12
,__inference_lstm_cell_2_layer_call_fn_307117
,__inference_lstm_cell_2_layer_call_fn_307134Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЕtrace_0zЖtrace_1

Зtrace_0
Иtrace_12в
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_307166
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_307198Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЗtrace_0zИtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
кBз
&__inference_dense_layer_call_fn_306753inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕBђ
A__inference_dense_layer_call_and_return_conditional_losses_306764inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рBн
,__inference_fingerprint_layer_call_fn_306773inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
G__inference_fingerprint_layer_call_and_return_conditional_losses_306784inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
мBй
(__inference_dense_1_layer_call_fn_306793inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_dense_1_layer_call_and_return_conditional_losses_306804inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
мBй
(__inference_dense_3_layer_call_fn_306813inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_dense_3_layer_call_and_return_conditional_losses_306824inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
мBй
(__inference_dense_2_layer_call_fn_306833inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_dense_2_layer_call_and_return_conditional_losses_306844inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
мBй
(__inference_dense_4_layer_call_fn_306853inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_dense_4_layer_call_and_return_conditional_losses_306864inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBо
-__inference_class_output_layer_call_fn_306873inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
H__inference_class_output_layer_call_and_return_conditional_losses_306884inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
фBс
0__inference_identity_output_layer_call_fn_306893inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
K__inference_identity_output_layer_call_and_return_conditional_losses_306904inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
Й	variables
К	keras_api

Лtotal

Мcount"
_tf_keras_metric
R
Н	variables
О	keras_api

Пtotal

Рcount"
_tf_keras_metric
R
С	variables
Т	keras_api

Уtotal

Фcount"
_tf_keras_metric
c
Х	variables
Ц	keras_api

Чtotal

Шcount
Щ
_fn_kwargs"
_tf_keras_metric
c
Ъ	variables
Ы	keras_api

Ьtotal

Эcount
Ю
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
*__inference_lstm_cell_layer_call_fn_306921inputsstates/0states/1"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
*__inference_lstm_cell_layer_call_fn_306938inputsstates/0states/1"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЈBЅ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_306970inputsstates/0states/1"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЈBЅ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_307002inputsstates/0states/1"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_lstm_cell_1_layer_call_fn_307019inputsstates/0states/1"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
,__inference_lstm_cell_1_layer_call_fn_307036inputsstates/0states/1"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЊBЇ
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_307068inputsstates/0states/1"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЊBЇ
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_307100inputsstates/0states/1"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_lstm_cell_2_layer_call_fn_307117inputsstates/0states/1"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
,__inference_lstm_cell_2_layer_call_fn_307134inputsstates/0states/1"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЊBЇ
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_307166inputsstates/0states/1"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЊBЇ
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_307198inputsstates/0states/1"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
Л0
М1"
trackable_list_wrapper
.
Й	variables"
_generic_user_object
:  (2total
:  (2count
0
П0
Р1"
trackable_list_wrapper
.
Н	variables"
_generic_user_object
:  (2total
:  (2count
0
У0
Ф1"
trackable_list_wrapper
.
С	variables"
_generic_user_object
:  (2total
:  (2count
0
Ч0
Ш1"
trackable_list_wrapper
.
Х	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ь0
Э1"
trackable_list_wrapper
.
Ъ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
#:!

2Adam/dense/kernel/m
:
2Adam/dense/bias/m
):'
@2Adam/fingerprint/kernel/m
#:!@2Adam/fingerprint/bias/m
%:#@ 2Adam/dense_1/kernel/m
: 2Adam/dense_1/bias/m
%:#@ 2Adam/dense_3/kernel/m
: 2Adam/dense_3/bias/m
%:# 2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
%:# 2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
*:(2Adam/class_output/kernel/m
$:"2Adam/class_output/bias/m
-:+2Adam/identity_output/kernel/m
':%2Adam/identity_output/bias/m
,:*Ex2Adam/lstm/lstm_cell/kernel/m
6:4x2&Adam/lstm/lstm_cell/recurrent_kernel/m
&:$x2Adam/lstm/lstm_cell/bias/m
0:.P2 Adam/lstm_1/lstm_cell_1/kernel/m
::8P2*Adam/lstm_1/lstm_cell_1/recurrent_kernel/m
*:(P2Adam/lstm_1/lstm_cell_1/bias/m
0:.(2 Adam/lstm_2/lstm_cell_2/kernel/m
::8
(2*Adam/lstm_2/lstm_cell_2/recurrent_kernel/m
*:((2Adam/lstm_2/lstm_cell_2/bias/m
#:!

2Adam/dense/kernel/v
:
2Adam/dense/bias/v
):'
@2Adam/fingerprint/kernel/v
#:!@2Adam/fingerprint/bias/v
%:#@ 2Adam/dense_1/kernel/v
: 2Adam/dense_1/bias/v
%:#@ 2Adam/dense_3/kernel/v
: 2Adam/dense_3/bias/v
%:# 2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
%:# 2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/v
*:(2Adam/class_output/kernel/v
$:"2Adam/class_output/bias/v
-:+2Adam/identity_output/kernel/v
':%2Adam/identity_output/bias/v
,:*Ex2Adam/lstm/lstm_cell/kernel/v
6:4x2&Adam/lstm/lstm_cell/recurrent_kernel/v
&:$x2Adam/lstm/lstm_cell/bias/v
0:.P2 Adam/lstm_1/lstm_cell_1/kernel/v
::8P2*Adam/lstm_1/lstm_cell_1/recurrent_kernel/v
*:(P2Adam/lstm_1/lstm_cell_1/bias/v
0:.(2 Adam/lstm_2/lstm_cell_2/kernel/v
::8
(2*Adam/lstm_2/lstm_cell_2/recurrent_kernel/v
*:((2Adam/lstm_2/lstm_cell_2/bias/vѕ
!__inference__wrapped_model_301028Яz{|}~@AHIXYPQhi`axypq4Ђ1
*Ђ'
%"
input_1џџџџџџџџџE
Њ "yЊv
6
class_output&#
class_outputџџџџџџџџџ
<
identity_output)&
identity_outputџџџџџџџџџЈ
H__inference_class_output_layer_call_and_return_conditional_losses_306884\pq/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
-__inference_class_output_layer_call_fn_306873Opq/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЃ
C__inference_dense_1_layer_call_and_return_conditional_losses_306804\PQ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ 
 {
(__inference_dense_1_layer_call_fn_306793OPQ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ Ѓ
C__inference_dense_2_layer_call_and_return_conditional_losses_306844\`a/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 {
(__inference_dense_2_layer_call_fn_306833O`a/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџЃ
C__inference_dense_3_layer_call_and_return_conditional_losses_306824\XY/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ 
 {
(__inference_dense_3_layer_call_fn_306813OXY/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ Ѓ
C__inference_dense_4_layer_call_and_return_conditional_losses_306864\hi/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 {
(__inference_dense_4_layer_call_fn_306853Ohi/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџЁ
A__inference_dense_layer_call_and_return_conditional_losses_306764\@A/Ђ,
%Ђ"
 
inputsџџџџџџџџџ

Њ "%Ђ"

0џџџџџџџџџ

 y
&__inference_dense_layer_call_fn_306753O@A/Ђ,
%Ђ"
 
inputsџџџџџџџџџ

Њ "џџџџџџџџџ
Ћ
C__inference_dropout_layer_call_and_return_conditional_losses_305492d7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ
p 
Њ ")Ђ&

0џџџџџџџџџ
 Ћ
C__inference_dropout_layer_call_and_return_conditional_losses_305504d7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ
p
Њ ")Ђ&

0џџџџџџџџџ
 
(__inference_dropout_layer_call_fn_305482W7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџ
(__inference_dropout_layer_call_fn_305487W7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџЇ
G__inference_fingerprint_layer_call_and_return_conditional_losses_306784\HI/Ђ,
%Ђ"
 
inputsџџџџџџџџџ

Њ "%Ђ"

0џџџџџџџџџ@
 
,__inference_fingerprint_layer_call_fn_306773OHI/Ђ,
%Ђ"
 
inputsџџџџџџџџџ

Њ "џџџџџџџџџ@Ћ
K__inference_identity_output_layer_call_and_return_conditional_losses_306904\xy/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
0__inference_identity_output_layer_call_fn_306893Oxy/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџб
B__inference_lstm_1_layer_call_and_return_conditional_losses_305691}~OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 б
B__inference_lstm_1_layer_call_and_return_conditional_losses_305834}~OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 З
B__inference_lstm_1_layer_call_and_return_conditional_losses_305977q}~?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 З
B__inference_lstm_1_layer_call_and_return_conditional_losses_306120q}~?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ ")Ђ&

0џџџџџџџџџ
 Ј
'__inference_lstm_1_layer_call_fn_305515}}~OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџЈ
'__inference_lstm_1_layer_call_fn_305526}}~OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%"џџџџџџџџџџџџџџџџџџ
'__inference_lstm_1_layer_call_fn_305537d}~?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ
'__inference_lstm_1_layer_call_fn_305548d}~?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџЧ
B__inference_lstm_2_layer_call_and_return_conditional_losses_306309OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 Ч
B__inference_lstm_2_layer_call_and_return_conditional_losses_306454OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ

 Ж
B__inference_lstm_2_layer_call_and_return_conditional_losses_306599p?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 Ж
B__inference_lstm_2_layer_call_and_return_conditional_losses_306744p?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ

 
'__inference_lstm_2_layer_call_fn_306131sOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ

'__inference_lstm_2_layer_call_fn_306142sOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ

'__inference_lstm_2_layer_call_fn_306153c?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ

'__inference_lstm_2_layer_call_fn_306164c?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ
Щ
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_307068§}~Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ
"
states/1џџџџџџџџџ
p 
Њ "sЂp
iЂf

0/0џџџџџџџџџ
EB

0/1/0џџџџџџџџџ

0/1/1џџџџџџџџџ
 Щ
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_307100§}~Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ
"
states/1џџџџџџџџџ
p
Њ "sЂp
iЂf

0/0џџџџџџџџџ
EB

0/1/0џџџџџџџџџ

0/1/1џџџџџџџџџ
 
,__inference_lstm_cell_1_layer_call_fn_307019э}~Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ
"
states/1џџџџџџџџџ
p 
Њ "cЂ`

0џџџџџџџџџ
A>

1/0џџџџџџџџџ

1/1џџџџџџџџџ
,__inference_lstm_cell_1_layer_call_fn_307036э}~Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ
"
states/1џџџџџџџџџ
p
Њ "cЂ`

0џџџџџџџџџ
A>

1/0џџџџџџџџџ

1/1џџџџџџџџџЬ
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_307166Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ

"
states/1џџџџџџџџџ

p 
Њ "sЂp
iЂf

0/0џџџџџџџџџ

EB

0/1/0џџџџџџџџџ


0/1/1џџџџџџџџџ

 Ь
G__inference_lstm_cell_2_layer_call_and_return_conditional_losses_307198Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ

"
states/1џџџџџџџџџ

p
Њ "sЂp
iЂf

0/0џџџџџџџџџ

EB

0/1/0џџџџџџџџџ


0/1/1џџџџџџџџџ

 Ё
,__inference_lstm_cell_2_layer_call_fn_307117№Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ

"
states/1џџџџџџџџџ

p 
Њ "cЂ`

0џџџџџџџџџ

A>

1/0џџџџџџџџџ


1/1џџџџџџџџџ
Ё
,__inference_lstm_cell_2_layer_call_fn_307134№Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ

"
states/1џџџџџџџџџ

p
Њ "cЂ`

0џџџџџџџџџ

A>

1/0џџџџџџџџџ


1/1џџџџџџџџџ
Ч
E__inference_lstm_cell_layer_call_and_return_conditional_losses_306970§z{|Ђ}
vЂs
 
inputsџџџџџџџџџE
KЂH
"
states/0џџџџџџџџџ
"
states/1џџџџџџџџџ
p 
Њ "sЂp
iЂf

0/0џџџџџџџџџ
EB

0/1/0џџџџџџџџџ

0/1/1џџџџџџџџџ
 Ч
E__inference_lstm_cell_layer_call_and_return_conditional_losses_307002§z{|Ђ}
vЂs
 
inputsџџџџџџџџџE
KЂH
"
states/0џџџџџџџџџ
"
states/1џџџџџџџџџ
p
Њ "sЂp
iЂf

0/0џџџџџџџџџ
EB

0/1/0џџџџџџџџџ

0/1/1џџџџџџџџџ
 
*__inference_lstm_cell_layer_call_fn_306921эz{|Ђ}
vЂs
 
inputsџџџџџџџџџE
KЂH
"
states/0џџџџџџџџџ
"
states/1џџџџџџџџџ
p 
Њ "cЂ`

0џџџџџџџџџ
A>

1/0џџџџџџџџџ

1/1џџџџџџџџџ
*__inference_lstm_cell_layer_call_fn_306938эz{|Ђ}
vЂs
 
inputsџџџџџџџџџE
KЂH
"
states/0џџџџџџџџџ
"
states/1џџџџџџџџџ
p
Њ "cЂ`

0џџџџџџџџџ
A>

1/0џџџџџџџџџ

1/1џџџџџџџџџЯ
@__inference_lstm_layer_call_and_return_conditional_losses_305048z{|OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџE

 
p 

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 Я
@__inference_lstm_layer_call_and_return_conditional_losses_305191z{|OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџE

 
p

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 Е
@__inference_lstm_layer_call_and_return_conditional_losses_305334qz{|?Ђ<
5Ђ2
$!
inputsџџџџџџџџџE

 
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 Е
@__inference_lstm_layer_call_and_return_conditional_losses_305477qz{|?Ђ<
5Ђ2
$!
inputsџџџџџџџџџE

 
p

 
Њ ")Ђ&

0џџџџџџџџџ
 І
%__inference_lstm_layer_call_fn_304872}z{|OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџE

 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџІ
%__inference_lstm_layer_call_fn_304883}z{|OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџE

 
p

 
Њ "%"џџџџџџџџџџџџџџџџџџ
%__inference_lstm_layer_call_fn_304894dz{|?Ђ<
5Ђ2
$!
inputsџџџџџџџџџE

 
p 

 
Њ "џџџџџџџџџ
%__inference_lstm_layer_call_fn_304905dz{|?Ђ<
5Ђ2
$!
inputsџџџџџџџџџE

 
p

 
Њ "џџџџџџџџџя
A__inference_model_layer_call_and_return_conditional_losses_303646Љz{|}~@AHIXYPQhi`axypq<Ђ9
2Ђ/
%"
input_1џџџџџџџџџE
p 

 
Њ "KЂH
A>

0/0џџџџџџџџџ

0/1џџџџџџџџџ
 я
A__inference_model_layer_call_and_return_conditional_losses_303713Љz{|}~@AHIXYPQhi`axypq<Ђ9
2Ђ/
%"
input_1џџџџџџџџџE
p

 
Њ "KЂH
A>

0/0џџџџџџџџџ

0/1џџџџџџџџџ
 ю
A__inference_model_layer_call_and_return_conditional_losses_304373Јz{|}~@AHIXYPQhi`axypq;Ђ8
1Ђ.
$!
inputsџџџџџџџџџE
p 

 
Њ "KЂH
A>

0/0џџџџџџџџџ

0/1џџџџџџџџџ
 ю
A__inference_model_layer_call_and_return_conditional_losses_304861Јz{|}~@AHIXYPQhi`axypq;Ђ8
1Ђ.
$!
inputsџџџџџџџџџE
p

 
Њ "KЂH
A>

0/0џџџџџџџџџ

0/1џџџџџџџџџ
 Ц
&__inference_model_layer_call_fn_302741z{|}~@AHIXYPQhi`axypq<Ђ9
2Ђ/
%"
input_1џџџџџџџџџE
p 

 
Њ "=:

0џџџџџџџџџ

1џџџџџџџџџЦ
&__inference_model_layer_call_fn_303579z{|}~@AHIXYPQhi`axypq<Ђ9
2Ђ/
%"
input_1џџџџџџџџџE
p

 
Њ "=:

0џџџџџџџџџ

1џџџџџџџџџХ
&__inference_model_layer_call_fn_303835z{|}~@AHIXYPQhi`axypq;Ђ8
1Ђ.
$!
inputsџџџџџџџџџE
p 

 
Њ "=:

0џџџџџџџџџ

1џџџџџџџџџХ
&__inference_model_layer_call_fn_303892z{|}~@AHIXYPQhi`axypq;Ђ8
1Ђ.
$!
inputsџџџџџџџџџE
p

 
Њ "=:

0џџџџџџџџџ

1џџџџџџџџџ
$__inference_signature_wrapper_303778кz{|}~@AHIXYPQhi`axypq?Ђ<
Ђ 
5Њ2
0
input_1%"
input_1џџџџџџџџџE"yЊv
6
class_output&#
class_outputџџџџџџџџџ
<
identity_output)&
identity_outputџџџџџџџџџ