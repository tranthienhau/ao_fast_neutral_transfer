??
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

B
Equal
x"T
y"T
z
"
Ttype:
2	
?
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
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
	MirrorPad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	"&
modestring:
REFLECT	SYMMETRIC
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
x
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2		"
align_cornersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
f

SaveSlices
filename
tensor_names
shapes_and_slices	
data2T"
T
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
:
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*1.13.12b'v1.13.0-rc2-5-g6612da8951'??
j
inputPlaceholder*(
_output_shapes
:??*
dtype0*
shape:??
?
MirrorPad/paddingsConst*9
value0B."         
   
   
   
           *
_output_shapes

:*
dtype0
?
	MirrorPad	MirrorPadinputMirrorPad/paddings*
T0*
mode	REFLECT*(
_output_shapes
:??*
	Tpaddings0
z
!conv1/conv/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"	   	          
e
 conv1/conv/truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
g
"conv1/conv/truncated_normal/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
+conv1/conv/truncated_normal/TruncatedNormalTruncatedNormal!conv1/conv/truncated_normal/shape*
dtype0*
T0*
seed2 *&
_output_shapes
:		 *

seed 
?
conv1/conv/truncated_normal/mulMul+conv1/conv/truncated_normal/TruncatedNormal"conv1/conv/truncated_normal/stddev*&
_output_shapes
:		 *
T0
?
conv1/conv/truncated_normalAddconv1/conv/truncated_normal/mul conv1/conv/truncated_normal/mean*&
_output_shapes
:		 *
T0
?
conv1/conv/weight
VariableV2*
dtype0*
	container *&
_output_shapes
:		 *
shared_name *
shape:		 
?
conv1/conv/weight/AssignAssignconv1/conv/weightconv1/conv/truncated_normal*
use_locking(*
validate_shape(*&
_output_shapes
:		 *$
_class
loc:@conv1/conv/weight*
T0
?
conv1/conv/weight/readIdentityconv1/conv/weight*&
_output_shapes
:		 *
T0*$
_class
loc:@conv1/conv/weight
?
conv1/conv/MirrorPad/paddingsConst*
dtype0*9
value0B."                             *
_output_shapes

:
?
conv1/conv/MirrorPad	MirrorPad	MirrorPadconv1/conv/MirrorPad/paddings*
T0*
	Tpaddings0*(
_output_shapes
:??*
mode	REFLECT
?
conv1/conv/convConv2Dconv1/conv/MirrorPadconv1/conv/weight/read*
	dilations
*
strides
*(
_output_shapes
:?? *
data_formatNHWC*
T0*
paddingVALID*
use_cudnn_on_gpu(
u
$conv1/moments/mean/reduction_indicesConst*
dtype0*
valueB"      *
_output_shapes
:
?
conv1/moments/meanMeanconv1/conv/conv$conv1/moments/mean/reduction_indices*
T0*
	keep_dims(*

Tidx0*&
_output_shapes
: 
o
conv1/moments/StopGradientStopGradientconv1/moments/mean*&
_output_shapes
: *
T0
?
conv1/moments/SquaredDifferenceSquaredDifferenceconv1/conv/convconv1/moments/StopGradient*(
_output_shapes
:?? *
T0
y
(conv1/moments/variance/reduction_indicesConst*
dtype0*
valueB"      *
_output_shapes
:
?
conv1/moments/varianceMeanconv1/moments/SquaredDifference(conv1/moments/variance/reduction_indices*
T0*
	keep_dims(*

Tidx0*&
_output_shapes
: 
h
	conv1/SubSubconv1/conv/convconv1/moments/mean*(
_output_shapes
:?? *
T0
P
conv1/Add/yConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0
f
	conv1/AddAddconv1/moments/varianceconv1/Add/y*&
_output_shapes
: *
T0
N

conv1/SqrtSqrt	conv1/Add*
T0*&
_output_shapes
: 
^
	conv1/divRealDiv	conv1/Sub
conv1/Sqrt*(
_output_shapes
:?? *
T0
P

conv1/ReluRelu	conv1/div*
T0*(
_output_shapes
:?? 
_
conv1/EqualEqual
conv1/Relu
conv1/Relu*(
_output_shapes
:?? *
T0
y
 conv1/zeros_like/shape_as_tensorConst*%
valueB"   ?  ?      *
dtype0*
_output_shapes
:
[
conv1/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
?
conv1/zeros_likeFill conv1/zeros_like/shape_as_tensorconv1/zeros_like/Const*(
_output_shapes
:?? *
T0*

index_type0
t
conv1/SelectSelectconv1/Equal
conv1/Reluconv1/zeros_like*
T0*(
_output_shapes
:?? 
z
!conv2/conv/truncated_normal/shapeConst*
dtype0*%
valueB"          @   *
_output_shapes
:
e
 conv2/conv/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
g
"conv2/conv/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *???=
?
+conv2/conv/truncated_normal/TruncatedNormalTruncatedNormal!conv2/conv/truncated_normal/shape*
seed2 *&
_output_shapes
: @*

seed *
dtype0*
T0
?
conv2/conv/truncated_normal/mulMul+conv2/conv/truncated_normal/TruncatedNormal"conv2/conv/truncated_normal/stddev*&
_output_shapes
: @*
T0
?
conv2/conv/truncated_normalAddconv2/conv/truncated_normal/mul conv2/conv/truncated_normal/mean*&
_output_shapes
: @*
T0
?
conv2/conv/weight
VariableV2*
dtype0*
shared_name *
shape: @*
	container *&
_output_shapes
: @
?
conv2/conv/weight/AssignAssignconv2/conv/weightconv2/conv/truncated_normal*
use_locking(*
validate_shape(*
T0*&
_output_shapes
: @*$
_class
loc:@conv2/conv/weight
?
conv2/conv/weight/readIdentityconv2/conv/weight*&
_output_shapes
: @*
T0*$
_class
loc:@conv2/conv/weight
?
conv2/conv/MirrorPad/paddingsConst*9
value0B."                             *
_output_shapes

:*
dtype0
?
conv2/conv/MirrorPad	MirrorPadconv1/Selectconv2/conv/MirrorPad/paddings*
mode	REFLECT*(
_output_shapes
:?? *
T0*
	Tpaddings0
?
conv2/conv/convConv2Dconv2/conv/MirrorPadconv2/conv/weight/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*
	dilations
*(
_output_shapes
:??@
u
$conv2/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB"      *
dtype0
?
conv2/moments/meanMeanconv2/conv/conv$conv2/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0*&
_output_shapes
:@
o
conv2/moments/StopGradientStopGradientconv2/moments/mean*&
_output_shapes
:@*
T0
?
conv2/moments/SquaredDifferenceSquaredDifferenceconv2/conv/convconv2/moments/StopGradient*
T0*(
_output_shapes
:??@
y
(conv2/moments/variance/reduction_indicesConst*
dtype0*
valueB"      *
_output_shapes
:
?
conv2/moments/varianceMeanconv2/moments/SquaredDifference(conv2/moments/variance/reduction_indices*&
_output_shapes
:@*
	keep_dims(*

Tidx0*
T0
h
	conv2/SubSubconv2/conv/convconv2/moments/mean*(
_output_shapes
:??@*
T0
P
conv2/Add/yConst*
dtype0*
_output_shapes
: *
valueB
 *_p?0
f
	conv2/AddAddconv2/moments/varianceconv2/Add/y*
T0*&
_output_shapes
:@
N

conv2/SqrtSqrt	conv2/Add*
T0*&
_output_shapes
:@
^
	conv2/divRealDiv	conv2/Sub
conv2/Sqrt*(
_output_shapes
:??@*
T0
P

conv2/ReluRelu	conv2/div*
T0*(
_output_shapes
:??@
_
conv2/EqualEqual
conv2/Relu
conv2/Relu*
T0*(
_output_shapes
:??@
y
 conv2/zeros_like/shape_as_tensorConst*
_output_shapes
:*%
valueB"   ?   ?   @   *
dtype0
[
conv2/zeros_like/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
?
conv2/zeros_likeFill conv2/zeros_like/shape_as_tensorconv2/zeros_like/Const*
T0*

index_type0*(
_output_shapes
:??@
t
conv2/SelectSelectconv2/Equal
conv2/Reluconv2/zeros_like*
T0*(
_output_shapes
:??@
z
!conv3/conv/truncated_normal/shapeConst*%
valueB"      @   ?   *
_output_shapes
:*
dtype0
e
 conv3/conv/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
g
"conv3/conv/truncated_normal/stddevConst*
valueB
 *???=*
_output_shapes
: *
dtype0
?
+conv3/conv/truncated_normal/TruncatedNormalTruncatedNormal!conv3/conv/truncated_normal/shape*
seed2 *
T0*'
_output_shapes
:@?*

seed *
dtype0
?
conv3/conv/truncated_normal/mulMul+conv3/conv/truncated_normal/TruncatedNormal"conv3/conv/truncated_normal/stddev*
T0*'
_output_shapes
:@?
?
conv3/conv/truncated_normalAddconv3/conv/truncated_normal/mul conv3/conv/truncated_normal/mean*
T0*'
_output_shapes
:@?
?
conv3/conv/weight
VariableV2*
shape:@?*
dtype0*'
_output_shapes
:@?*
shared_name *
	container 
?
conv3/conv/weight/AssignAssignconv3/conv/weightconv3/conv/truncated_normal*'
_output_shapes
:@?*
use_locking(*$
_class
loc:@conv3/conv/weight*
validate_shape(*
T0
?
conv3/conv/weight/readIdentityconv3/conv/weight*
T0*$
_class
loc:@conv3/conv/weight*'
_output_shapes
:@?
?
conv3/conv/MirrorPad/paddingsConst*
dtype0*9
value0B."                             *
_output_shapes

:
?
conv3/conv/MirrorPad	MirrorPadconv2/Selectconv3/conv/MirrorPad/paddings*
mode	REFLECT*
T0*
	Tpaddings0*(
_output_shapes
:??@
?
conv3/conv/convConv2Dconv3/conv/MirrorPadconv3/conv/weight/read*'
_output_shapes
:ii?*
	dilations
*
data_formatNHWC*
strides
*
paddingVALID*
T0*
use_cudnn_on_gpu(
u
$conv3/moments/mean/reduction_indicesConst*
dtype0*
valueB"      *
_output_shapes
:
?
conv3/moments/meanMeanconv3/conv/conv$conv3/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*'
_output_shapes
:?*
T0
p
conv3/moments/StopGradientStopGradientconv3/moments/mean*
T0*'
_output_shapes
:?
?
conv3/moments/SquaredDifferenceSquaredDifferenceconv3/conv/convconv3/moments/StopGradient*'
_output_shapes
:ii?*
T0
y
(conv3/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB"      *
dtype0
?
conv3/moments/varianceMeanconv3/moments/SquaredDifference(conv3/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:?
g
	conv3/SubSubconv3/conv/convconv3/moments/mean*'
_output_shapes
:ii?*
T0
P
conv3/Add/yConst*
_output_shapes
: *
valueB
 *_p?0*
dtype0
g
	conv3/AddAddconv3/moments/varianceconv3/Add/y*'
_output_shapes
:?*
T0
O

conv3/SqrtSqrt	conv3/Add*
T0*'
_output_shapes
:?
]
	conv3/divRealDiv	conv3/Sub
conv3/Sqrt*'
_output_shapes
:ii?*
T0
O

conv3/ReluRelu	conv3/div*'
_output_shapes
:ii?*
T0
^
conv3/EqualEqual
conv3/Relu
conv3/Relu*'
_output_shapes
:ii?*
T0
y
 conv3/zeros_like/shape_as_tensorConst*%
valueB"   i   i   ?   *
dtype0*
_output_shapes
:
[
conv3/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
?
conv3/zeros_likeFill conv3/zeros_like/shape_as_tensorconv3/zeros_like/Const*

index_type0*
T0*'
_output_shapes
:ii?
s
conv3/SelectSelectconv3/Equal
conv3/Reluconv3/zeros_like*'
_output_shapes
:ii?*
T0
?
)res1/residual/conv/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   
m
(res1/residual/conv/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
o
*res1/residual/conv/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=
?
3res1/residual/conv/truncated_normal/TruncatedNormalTruncatedNormal)res1/residual/conv/truncated_normal/shape*
seed2 *
T0*

seed *
dtype0*(
_output_shapes
:??
?
'res1/residual/conv/truncated_normal/mulMul3res1/residual/conv/truncated_normal/TruncatedNormal*res1/residual/conv/truncated_normal/stddev*
T0*(
_output_shapes
:??
?
#res1/residual/conv/truncated_normalAdd'res1/residual/conv/truncated_normal/mul(res1/residual/conv/truncated_normal/mean*(
_output_shapes
:??*
T0
?
res1/residual/conv/weight
VariableV2*(
_output_shapes
:??*
dtype0*
shape:??*
shared_name *
	container 
?
 res1/residual/conv/weight/AssignAssignres1/residual/conv/weight#res1/residual/conv/truncated_normal*,
_class"
 loc:@res1/residual/conv/weight*(
_output_shapes
:??*
T0*
use_locking(*
validate_shape(
?
res1/residual/conv/weight/readIdentityres1/residual/conv/weight*,
_class"
 loc:@res1/residual/conv/weight*
T0*(
_output_shapes
:??
?
%res1/residual/conv/MirrorPad/paddingsConst*
dtype0*9
value0B."                             *
_output_shapes

:
?
res1/residual/conv/MirrorPad	MirrorPadconv3/Select%res1/residual/conv/MirrorPad/paddings*
	Tpaddings0*'
_output_shapes
:kk?*
T0*
mode	REFLECT
?
res1/residual/conv/convConv2Dres1/residual/conv/MirrorPadres1/residual/conv/weight/read*'
_output_shapes
:ii?*
strides
*
use_cudnn_on_gpu(*
data_formatNHWC*
T0*
	dilations
*
paddingVALID
e
res1/residual/ReluRelures1/residual/conv/conv*
T0*'
_output_shapes
:ii?
v
res1/residual/EqualEqualres1/residual/Relures1/residual/Relu*
T0*'
_output_shapes
:ii?
?
(res1/residual/zeros_like/shape_as_tensorConst*%
valueB"   i   i   ?   *
dtype0*
_output_shapes
:
c
res1/residual/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
?
res1/residual/zeros_likeFill(res1/residual/zeros_like/shape_as_tensorres1/residual/zeros_like/Const*

index_type0*
T0*'
_output_shapes
:ii?
?
res1/residual/SelectSelectres1/residual/Equalres1/residual/Relures1/residual/zeros_like*'
_output_shapes
:ii?*
T0
?
+res1/residual/conv_1/truncated_normal/shapeConst*%
valueB"      ?   ?   *
_output_shapes
:*
dtype0
o
*res1/residual/conv_1/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
,res1/residual/conv_1/truncated_normal/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
5res1/residual/conv_1/truncated_normal/TruncatedNormalTruncatedNormal+res1/residual/conv_1/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:??
?
)res1/residual/conv_1/truncated_normal/mulMul5res1/residual/conv_1/truncated_normal/TruncatedNormal,res1/residual/conv_1/truncated_normal/stddev*
T0*(
_output_shapes
:??
?
%res1/residual/conv_1/truncated_normalAdd)res1/residual/conv_1/truncated_normal/mul*res1/residual/conv_1/truncated_normal/mean*(
_output_shapes
:??*
T0
?
res1/residual/conv_1/weight
VariableV2*
dtype0*
	container *(
_output_shapes
:??*
shape:??*
shared_name 
?
"res1/residual/conv_1/weight/AssignAssignres1/residual/conv_1/weight%res1/residual/conv_1/truncated_normal*
use_locking(*(
_output_shapes
:??*.
_class$
" loc:@res1/residual/conv_1/weight*
T0*
validate_shape(
?
 res1/residual/conv_1/weight/readIdentityres1/residual/conv_1/weight*.
_class$
" loc:@res1/residual/conv_1/weight*(
_output_shapes
:??*
T0
?
'res1/residual/conv_1/MirrorPad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
?
res1/residual/conv_1/MirrorPad	MirrorPadres1/residual/Select'res1/residual/conv_1/MirrorPad/paddings*
	Tpaddings0*'
_output_shapes
:kk?*
mode	REFLECT*
T0
?
res1/residual/conv_1/convConv2Dres1/residual/conv_1/MirrorPad res1/residual/conv_1/weight/read*
	dilations
*
strides
*'
_output_shapes
:ii?*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
s
res1/residual/addAddconv3/Selectres1/residual/conv_1/conv*
T0*'
_output_shapes
:ii?
?
)res2/residual/conv/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"      ?   ?   *
dtype0
m
(res2/residual/conv/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
o
*res2/residual/conv/truncated_normal/stddevConst*
valueB
 *???=*
_output_shapes
: *
dtype0
?
3res2/residual/conv/truncated_normal/TruncatedNormalTruncatedNormal)res2/residual/conv/truncated_normal/shape*(
_output_shapes
:??*

seed *
seed2 *
T0*
dtype0
?
'res2/residual/conv/truncated_normal/mulMul3res2/residual/conv/truncated_normal/TruncatedNormal*res2/residual/conv/truncated_normal/stddev*
T0*(
_output_shapes
:??
?
#res2/residual/conv/truncated_normalAdd'res2/residual/conv/truncated_normal/mul(res2/residual/conv/truncated_normal/mean*(
_output_shapes
:??*
T0
?
res2/residual/conv/weight
VariableV2*(
_output_shapes
:??*
dtype0*
shape:??*
shared_name *
	container 
?
 res2/residual/conv/weight/AssignAssignres2/residual/conv/weight#res2/residual/conv/truncated_normal*
validate_shape(*,
_class"
 loc:@res2/residual/conv/weight*
use_locking(*
T0*(
_output_shapes
:??
?
res2/residual/conv/weight/readIdentityres2/residual/conv/weight*
T0*(
_output_shapes
:??*,
_class"
 loc:@res2/residual/conv/weight
?
%res2/residual/conv/MirrorPad/paddingsConst*
dtype0*9
value0B."                             *
_output_shapes

:
?
res2/residual/conv/MirrorPad	MirrorPadres1/residual/add%res2/residual/conv/MirrorPad/paddings*
	Tpaddings0*
mode	REFLECT*'
_output_shapes
:kk?*
T0
?
res2/residual/conv/convConv2Dres2/residual/conv/MirrorPadres2/residual/conv/weight/read*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*
	dilations
*
strides
*'
_output_shapes
:ii?
e
res2/residual/ReluRelures2/residual/conv/conv*'
_output_shapes
:ii?*
T0
v
res2/residual/EqualEqualres2/residual/Relures2/residual/Relu*'
_output_shapes
:ii?*
T0
?
(res2/residual/zeros_like/shape_as_tensorConst*
_output_shapes
:*%
valueB"   i   i   ?   *
dtype0
c
res2/residual/zeros_like/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
?
res2/residual/zeros_likeFill(res2/residual/zeros_like/shape_as_tensorres2/residual/zeros_like/Const*'
_output_shapes
:ii?*
T0*

index_type0
?
res2/residual/SelectSelectres2/residual/Equalres2/residual/Relures2/residual/zeros_like*
T0*'
_output_shapes
:ii?
?
+res2/residual/conv_1/truncated_normal/shapeConst*%
valueB"      ?   ?   *
_output_shapes
:*
dtype0
o
*res2/residual/conv_1/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
q
,res2/residual/conv_1/truncated_normal/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
5res2/residual/conv_1/truncated_normal/TruncatedNormalTruncatedNormal+res2/residual/conv_1/truncated_normal/shape*
seed2 *
T0*(
_output_shapes
:??*

seed *
dtype0
?
)res2/residual/conv_1/truncated_normal/mulMul5res2/residual/conv_1/truncated_normal/TruncatedNormal,res2/residual/conv_1/truncated_normal/stddev*(
_output_shapes
:??*
T0
?
%res2/residual/conv_1/truncated_normalAdd)res2/residual/conv_1/truncated_normal/mul*res2/residual/conv_1/truncated_normal/mean*(
_output_shapes
:??*
T0
?
res2/residual/conv_1/weight
VariableV2*
	container *
shared_name *(
_output_shapes
:??*
shape:??*
dtype0
?
"res2/residual/conv_1/weight/AssignAssignres2/residual/conv_1/weight%res2/residual/conv_1/truncated_normal*(
_output_shapes
:??*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@res2/residual/conv_1/weight
?
 res2/residual/conv_1/weight/readIdentityres2/residual/conv_1/weight*
T0*.
_class$
" loc:@res2/residual/conv_1/weight*(
_output_shapes
:??
?
'res2/residual/conv_1/MirrorPad/paddingsConst*
dtype0*9
value0B."                             *
_output_shapes

:
?
res2/residual/conv_1/MirrorPad	MirrorPadres2/residual/Select'res2/residual/conv_1/MirrorPad/paddings*
mode	REFLECT*
T0*'
_output_shapes
:kk?*
	Tpaddings0
?
res2/residual/conv_1/convConv2Dres2/residual/conv_1/MirrorPad res2/residual/conv_1/weight/read*
strides
*
T0*'
_output_shapes
:ii?*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
	dilations

x
res2/residual/addAddres1/residual/addres2/residual/conv_1/conv*
T0*'
_output_shapes
:ii?
?
)res3/residual/conv/truncated_normal/shapeConst*%
valueB"      ?   ?   *
dtype0*
_output_shapes
:
m
(res3/residual/conv/truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
o
*res3/residual/conv/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=
?
3res3/residual/conv/truncated_normal/TruncatedNormalTruncatedNormal)res3/residual/conv/truncated_normal/shape*
dtype0*

seed *
seed2 *(
_output_shapes
:??*
T0
?
'res3/residual/conv/truncated_normal/mulMul3res3/residual/conv/truncated_normal/TruncatedNormal*res3/residual/conv/truncated_normal/stddev*
T0*(
_output_shapes
:??
?
#res3/residual/conv/truncated_normalAdd'res3/residual/conv/truncated_normal/mul(res3/residual/conv/truncated_normal/mean*(
_output_shapes
:??*
T0
?
res3/residual/conv/weight
VariableV2*
	container *(
_output_shapes
:??*
dtype0*
shared_name *
shape:??
?
 res3/residual/conv/weight/AssignAssignres3/residual/conv/weight#res3/residual/conv/truncated_normal*(
_output_shapes
:??*
validate_shape(*
T0*,
_class"
 loc:@res3/residual/conv/weight*
use_locking(
?
res3/residual/conv/weight/readIdentityres3/residual/conv/weight*(
_output_shapes
:??*,
_class"
 loc:@res3/residual/conv/weight*
T0
?
%res3/residual/conv/MirrorPad/paddingsConst*
dtype0*9
value0B."                             *
_output_shapes

:
?
res3/residual/conv/MirrorPad	MirrorPadres2/residual/add%res3/residual/conv/MirrorPad/paddings*'
_output_shapes
:kk?*
	Tpaddings0*
T0*
mode	REFLECT
?
res3/residual/conv/convConv2Dres3/residual/conv/MirrorPadres3/residual/conv/weight/read*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:ii?*
data_formatNHWC*
	dilations
*
T0
e
res3/residual/ReluRelures3/residual/conv/conv*
T0*'
_output_shapes
:ii?
v
res3/residual/EqualEqualres3/residual/Relures3/residual/Relu*'
_output_shapes
:ii?*
T0
?
(res3/residual/zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"   i   i   ?   
c
res3/residual/zeros_like/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
?
res3/residual/zeros_likeFill(res3/residual/zeros_like/shape_as_tensorres3/residual/zeros_like/Const*

index_type0*
T0*'
_output_shapes
:ii?
?
res3/residual/SelectSelectres3/residual/Equalres3/residual/Relures3/residual/zeros_like*'
_output_shapes
:ii?*
T0
?
+res3/residual/conv_1/truncated_normal/shapeConst*%
valueB"      ?   ?   *
_output_shapes
:*
dtype0
o
*res3/residual/conv_1/truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
q
,res3/residual/conv_1/truncated_normal/stddevConst*
valueB
 *???=*
_output_shapes
: *
dtype0
?
5res3/residual/conv_1/truncated_normal/TruncatedNormalTruncatedNormal+res3/residual/conv_1/truncated_normal/shape*
dtype0*
T0*
seed2 *

seed *(
_output_shapes
:??
?
)res3/residual/conv_1/truncated_normal/mulMul5res3/residual/conv_1/truncated_normal/TruncatedNormal,res3/residual/conv_1/truncated_normal/stddev*
T0*(
_output_shapes
:??
?
%res3/residual/conv_1/truncated_normalAdd)res3/residual/conv_1/truncated_normal/mul*res3/residual/conv_1/truncated_normal/mean*
T0*(
_output_shapes
:??
?
res3/residual/conv_1/weight
VariableV2*
shared_name *
shape:??*
dtype0*
	container *(
_output_shapes
:??
?
"res3/residual/conv_1/weight/AssignAssignres3/residual/conv_1/weight%res3/residual/conv_1/truncated_normal*(
_output_shapes
:??*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@res3/residual/conv_1/weight
?
 res3/residual/conv_1/weight/readIdentityres3/residual/conv_1/weight*(
_output_shapes
:??*.
_class$
" loc:@res3/residual/conv_1/weight*
T0
?
'res3/residual/conv_1/MirrorPad/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             
?
res3/residual/conv_1/MirrorPad	MirrorPadres3/residual/Select'res3/residual/conv_1/MirrorPad/paddings*
mode	REFLECT*'
_output_shapes
:kk?*
	Tpaddings0*
T0
?
res3/residual/conv_1/convConv2Dres3/residual/conv_1/MirrorPad res3/residual/conv_1/weight/read*
T0*
data_formatNHWC*
	dilations
*
paddingVALID*
strides
*'
_output_shapes
:ii?*
use_cudnn_on_gpu(
x
res3/residual/addAddres2/residual/addres3/residual/conv_1/conv*
T0*'
_output_shapes
:ii?
?
)res4/residual/conv/truncated_normal/shapeConst*%
valueB"      ?   ?   *
_output_shapes
:*
dtype0
m
(res4/residual/conv/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
o
*res4/residual/conv/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *???=
?
3res4/residual/conv/truncated_normal/TruncatedNormalTruncatedNormal)res4/residual/conv/truncated_normal/shape*(
_output_shapes
:??*

seed *
T0*
dtype0*
seed2 
?
'res4/residual/conv/truncated_normal/mulMul3res4/residual/conv/truncated_normal/TruncatedNormal*res4/residual/conv/truncated_normal/stddev*(
_output_shapes
:??*
T0
?
#res4/residual/conv/truncated_normalAdd'res4/residual/conv/truncated_normal/mul(res4/residual/conv/truncated_normal/mean*
T0*(
_output_shapes
:??
?
res4/residual/conv/weight
VariableV2*
	container *
shared_name *
dtype0*
shape:??*(
_output_shapes
:??
?
 res4/residual/conv/weight/AssignAssignres4/residual/conv/weight#res4/residual/conv/truncated_normal*
validate_shape(*
use_locking(*
T0*(
_output_shapes
:??*,
_class"
 loc:@res4/residual/conv/weight
?
res4/residual/conv/weight/readIdentityres4/residual/conv/weight*
T0*,
_class"
 loc:@res4/residual/conv/weight*(
_output_shapes
:??
?
%res4/residual/conv/MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
?
res4/residual/conv/MirrorPad	MirrorPadres3/residual/add%res4/residual/conv/MirrorPad/paddings*'
_output_shapes
:kk?*
	Tpaddings0*
mode	REFLECT*
T0
?
res4/residual/conv/convConv2Dres4/residual/conv/MirrorPadres4/residual/conv/weight/read*
use_cudnn_on_gpu(*
	dilations
*'
_output_shapes
:ii?*
strides
*
T0*
paddingVALID*
data_formatNHWC
e
res4/residual/ReluRelures4/residual/conv/conv*'
_output_shapes
:ii?*
T0
v
res4/residual/EqualEqualres4/residual/Relures4/residual/Relu*'
_output_shapes
:ii?*
T0
?
(res4/residual/zeros_like/shape_as_tensorConst*%
valueB"   i   i   ?   *
dtype0*
_output_shapes
:
c
res4/residual/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
?
res4/residual/zeros_likeFill(res4/residual/zeros_like/shape_as_tensorres4/residual/zeros_like/Const*

index_type0*
T0*'
_output_shapes
:ii?
?
res4/residual/SelectSelectres4/residual/Equalres4/residual/Relures4/residual/zeros_like*'
_output_shapes
:ii?*
T0
?
+res4/residual/conv_1/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   
o
*res4/residual/conv_1/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
,res4/residual/conv_1/truncated_normal/stddevConst*
valueB
 *???=*
_output_shapes
: *
dtype0
?
5res4/residual/conv_1/truncated_normal/TruncatedNormalTruncatedNormal+res4/residual/conv_1/truncated_normal/shape*

seed *(
_output_shapes
:??*
T0*
seed2 *
dtype0
?
)res4/residual/conv_1/truncated_normal/mulMul5res4/residual/conv_1/truncated_normal/TruncatedNormal,res4/residual/conv_1/truncated_normal/stddev*(
_output_shapes
:??*
T0
?
%res4/residual/conv_1/truncated_normalAdd)res4/residual/conv_1/truncated_normal/mul*res4/residual/conv_1/truncated_normal/mean*
T0*(
_output_shapes
:??
?
res4/residual/conv_1/weight
VariableV2*
shared_name *
dtype0*
	container *
shape:??*(
_output_shapes
:??
?
"res4/residual/conv_1/weight/AssignAssignres4/residual/conv_1/weight%res4/residual/conv_1/truncated_normal*(
_output_shapes
:??*
validate_shape(*.
_class$
" loc:@res4/residual/conv_1/weight*
T0*
use_locking(
?
 res4/residual/conv_1/weight/readIdentityres4/residual/conv_1/weight*
T0*.
_class$
" loc:@res4/residual/conv_1/weight*(
_output_shapes
:??
?
'res4/residual/conv_1/MirrorPad/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             
?
res4/residual/conv_1/MirrorPad	MirrorPadres4/residual/Select'res4/residual/conv_1/MirrorPad/paddings*
T0*
mode	REFLECT*
	Tpaddings0*'
_output_shapes
:kk?
?
res4/residual/conv_1/convConv2Dres4/residual/conv_1/MirrorPad res4/residual/conv_1/weight/read*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
	dilations
*'
_output_shapes
:ii?*
T0*
paddingVALID
x
res4/residual/addAddres3/residual/addres4/residual/conv_1/conv*
T0*'
_output_shapes
:ii?
?
)res5/residual/conv/truncated_normal/shapeConst*%
valueB"      ?   ?   *
dtype0*
_output_shapes
:
m
(res5/residual/conv/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
o
*res5/residual/conv/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *???=*
dtype0
?
3res5/residual/conv/truncated_normal/TruncatedNormalTruncatedNormal)res5/residual/conv/truncated_normal/shape*

seed *(
_output_shapes
:??*
T0*
dtype0*
seed2 
?
'res5/residual/conv/truncated_normal/mulMul3res5/residual/conv/truncated_normal/TruncatedNormal*res5/residual/conv/truncated_normal/stddev*(
_output_shapes
:??*
T0
?
#res5/residual/conv/truncated_normalAdd'res5/residual/conv/truncated_normal/mul(res5/residual/conv/truncated_normal/mean*(
_output_shapes
:??*
T0
?
res5/residual/conv/weight
VariableV2*
dtype0*
	container *(
_output_shapes
:??*
shape:??*
shared_name 
?
 res5/residual/conv/weight/AssignAssignres5/residual/conv/weight#res5/residual/conv/truncated_normal*,
_class"
 loc:@res5/residual/conv/weight*
T0*(
_output_shapes
:??*
validate_shape(*
use_locking(
?
res5/residual/conv/weight/readIdentityres5/residual/conv/weight*
T0*,
_class"
 loc:@res5/residual/conv/weight*(
_output_shapes
:??
?
%res5/residual/conv/MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
?
res5/residual/conv/MirrorPad	MirrorPadres4/residual/add%res5/residual/conv/MirrorPad/paddings*
	Tpaddings0*'
_output_shapes
:kk?*
T0*
mode	REFLECT
?
res5/residual/conv/convConv2Dres5/residual/conv/MirrorPadres5/residual/conv/weight/read*
T0*'
_output_shapes
:ii?*
	dilations
*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
e
res5/residual/ReluRelures5/residual/conv/conv*'
_output_shapes
:ii?*
T0
v
res5/residual/EqualEqualres5/residual/Relures5/residual/Relu*'
_output_shapes
:ii?*
T0
?
(res5/residual/zeros_like/shape_as_tensorConst*%
valueB"   i   i   ?   *
dtype0*
_output_shapes
:
c
res5/residual/zeros_like/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
?
res5/residual/zeros_likeFill(res5/residual/zeros_like/shape_as_tensorres5/residual/zeros_like/Const*
T0*

index_type0*'
_output_shapes
:ii?
?
res5/residual/SelectSelectres5/residual/Equalres5/residual/Relures5/residual/zeros_like*'
_output_shapes
:ii?*
T0
?
+res5/residual/conv_1/truncated_normal/shapeConst*%
valueB"      ?   ?   *
dtype0*
_output_shapes
:
o
*res5/residual/conv_1/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
q
,res5/residual/conv_1/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=
?
5res5/residual/conv_1/truncated_normal/TruncatedNormalTruncatedNormal+res5/residual/conv_1/truncated_normal/shape*

seed *
dtype0*
T0*(
_output_shapes
:??*
seed2 
?
)res5/residual/conv_1/truncated_normal/mulMul5res5/residual/conv_1/truncated_normal/TruncatedNormal,res5/residual/conv_1/truncated_normal/stddev*
T0*(
_output_shapes
:??
?
%res5/residual/conv_1/truncated_normalAdd)res5/residual/conv_1/truncated_normal/mul*res5/residual/conv_1/truncated_normal/mean*
T0*(
_output_shapes
:??
?
res5/residual/conv_1/weight
VariableV2*
dtype0*(
_output_shapes
:??*
	container *
shared_name *
shape:??
?
"res5/residual/conv_1/weight/AssignAssignres5/residual/conv_1/weight%res5/residual/conv_1/truncated_normal*
use_locking(*
validate_shape(*(
_output_shapes
:??*.
_class$
" loc:@res5/residual/conv_1/weight*
T0
?
 res5/residual/conv_1/weight/readIdentityres5/residual/conv_1/weight*.
_class$
" loc:@res5/residual/conv_1/weight*
T0*(
_output_shapes
:??
?
'res5/residual/conv_1/MirrorPad/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             
?
res5/residual/conv_1/MirrorPad	MirrorPadres5/residual/Select'res5/residual/conv_1/MirrorPad/paddings*
	Tpaddings0*
T0*
mode	REFLECT*'
_output_shapes
:kk?
?
res5/residual/conv_1/convConv2Dres5/residual/conv_1/MirrorPad res5/residual/conv_1/weight/read*
T0*
use_cudnn_on_gpu(*'
_output_shapes
:ii?*
strides
*
	dilations
*
data_formatNHWC*
paddingVALID
x
res5/residual/addAddres4/residual/addres5/residual/conv_1/conv*'
_output_shapes
:ii?*
T0
u
deconv1/conv_transpose/ShapeConst*%
valueB"   i   i   ?   *
_output_shapes
:*
dtype0
t
*deconv1/conv_transpose/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
v
,deconv1/conv_transpose/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
v
,deconv1/conv_transpose/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
?
$deconv1/conv_transpose/strided_sliceStridedSlicedeconv1/conv_transpose/Shape*deconv1/conv_transpose/strided_slice/stack,deconv1/conv_transpose/strided_slice/stack_1,deconv1/conv_transpose/strided_slice/stack_2*
new_axis_mask *
end_mask *
shrink_axis_mask*
_output_shapes
: *
Index0*

begin_mask *
ellipsis_mask *
T0
w
deconv1/conv_transpose/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"   i   i   ?   
v
,deconv1/conv_transpose/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
x
.deconv1/conv_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
x
.deconv1/conv_transpose/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
&deconv1/conv_transpose/strided_slice_1StridedSlicedeconv1/conv_transpose/Shape_1,deconv1/conv_transpose/strided_slice_1/stack.deconv1/conv_transpose/strided_slice_1/stack_1.deconv1/conv_transpose/strided_slice_1/stack_2*
_output_shapes
: *
new_axis_mask *

begin_mask *
T0*
ellipsis_mask *
Index0*
shrink_axis_mask*
end_mask 
^
deconv1/conv_transpose/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
deconv1/conv_transpose/mulMul$deconv1/conv_transpose/strided_slicedeconv1/conv_transpose/mul/y*
_output_shapes
: *
T0
`
deconv1/conv_transpose/mul_1/yConst*
dtype0*
_output_shapes
: *
value	B :
?
deconv1/conv_transpose/mul_1Muldeconv1/conv_transpose/muldeconv1/conv_transpose/mul_1/y*
T0*
_output_shapes
: 
`
deconv1/conv_transpose/mul_2/yConst*
value	B :*
_output_shapes
: *
dtype0
?
deconv1/conv_transpose/mul_2Mul&deconv1/conv_transpose/strided_slice_1deconv1/conv_transpose/mul_2/y*
_output_shapes
: *
T0
`
deconv1/conv_transpose/mul_3/yConst*
_output_shapes
: *
dtype0*
value	B :
?
deconv1/conv_transpose/mul_3Muldeconv1/conv_transpose/mul_2deconv1/conv_transpose/mul_3/y*
T0*
_output_shapes
: 
?
"deconv1/conv_transpose/resize/sizePackdeconv1/conv_transpose/mul_1deconv1/conv_transpose/mul_3*
N*
_output_shapes
:*

axis *
T0
?
3deconv1/conv_transpose/resize/ResizeNearestNeighborResizeNearestNeighborres5/residual/add"deconv1/conv_transpose/resize/size*
align_corners( *
T0*)
_output_shapes
:???
?
2deconv1/conv_transpose/conv/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   @   
v
1deconv1/conv_transpose/conv/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
x
3deconv1/conv_transpose/conv/truncated_normal/stddevConst*
valueB
 *???=*
_output_shapes
: *
dtype0
?
<deconv1/conv_transpose/conv/truncated_normal/TruncatedNormalTruncatedNormal2deconv1/conv_transpose/conv/truncated_normal/shape*
seed2 *'
_output_shapes
:?@*
dtype0*
T0*

seed 
?
0deconv1/conv_transpose/conv/truncated_normal/mulMul<deconv1/conv_transpose/conv/truncated_normal/TruncatedNormal3deconv1/conv_transpose/conv/truncated_normal/stddev*'
_output_shapes
:?@*
T0
?
,deconv1/conv_transpose/conv/truncated_normalAdd0deconv1/conv_transpose/conv/truncated_normal/mul1deconv1/conv_transpose/conv/truncated_normal/mean*
T0*'
_output_shapes
:?@
?
"deconv1/conv_transpose/conv/weight
VariableV2*'
_output_shapes
:?@*
dtype0*
shape:?@*
	container *
shared_name 
?
)deconv1/conv_transpose/conv/weight/AssignAssign"deconv1/conv_transpose/conv/weight,deconv1/conv_transpose/conv/truncated_normal*
use_locking(*
T0*5
_class+
)'loc:@deconv1/conv_transpose/conv/weight*'
_output_shapes
:?@*
validate_shape(
?
'deconv1/conv_transpose/conv/weight/readIdentity"deconv1/conv_transpose/conv/weight*
T0*'
_output_shapes
:?@*5
_class+
)'loc:@deconv1/conv_transpose/conv/weight
?
.deconv1/conv_transpose/conv/MirrorPad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
?
%deconv1/conv_transpose/conv/MirrorPad	MirrorPad3deconv1/conv_transpose/resize/ResizeNearestNeighbor.deconv1/conv_transpose/conv/MirrorPad/paddings*
mode	REFLECT*
	Tpaddings0*
T0*)
_output_shapes
:???
?
 deconv1/conv_transpose/conv/convConv2D%deconv1/conv_transpose/conv/MirrorPad'deconv1/conv_transpose/conv/weight/read*
use_cudnn_on_gpu(*
paddingVALID*(
_output_shapes
:??@*
T0*
data_formatNHWC*
strides
*
	dilations

w
&deconv1/moments/mean/reduction_indicesConst*
dtype0*
valueB"      *
_output_shapes
:
?
deconv1/moments/meanMean deconv1/conv_transpose/conv/conv&deconv1/moments/mean/reduction_indices*
T0*&
_output_shapes
:@*

Tidx0*
	keep_dims(
s
deconv1/moments/StopGradientStopGradientdeconv1/moments/mean*
T0*&
_output_shapes
:@
?
!deconv1/moments/SquaredDifferenceSquaredDifference deconv1/conv_transpose/conv/convdeconv1/moments/StopGradient*(
_output_shapes
:??@*
T0
{
*deconv1/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB"      *
dtype0
?
deconv1/moments/varianceMean!deconv1/moments/SquaredDifference*deconv1/moments/variance/reduction_indices*&
_output_shapes
:@*
T0*

Tidx0*
	keep_dims(
}
deconv1/SubSub deconv1/conv_transpose/conv/convdeconv1/moments/mean*
T0*(
_output_shapes
:??@
R
deconv1/Add/yConst*
valueB
 *_p?0*
dtype0*
_output_shapes
: 
l
deconv1/AddAdddeconv1/moments/variancedeconv1/Add/y*
T0*&
_output_shapes
:@
R
deconv1/SqrtSqrtdeconv1/Add*&
_output_shapes
:@*
T0
d
deconv1/divRealDivdeconv1/Subdeconv1/Sqrt*(
_output_shapes
:??@*
T0
T
deconv1/ReluReludeconv1/div*
T0*(
_output_shapes
:??@
e
deconv1/EqualEqualdeconv1/Reludeconv1/Relu*
T0*(
_output_shapes
:??@
{
"deconv1/zeros_like/shape_as_tensorConst*%
valueB"   ?   ?   @   *
_output_shapes
:*
dtype0
]
deconv1/zeros_like/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
?
deconv1/zeros_likeFill"deconv1/zeros_like/shape_as_tensordeconv1/zeros_like/Const*
T0*(
_output_shapes
:??@*

index_type0
|
deconv1/SelectSelectdeconv1/Equaldeconv1/Reludeconv1/zeros_like*
T0*(
_output_shapes
:??@
u
deconv2/conv_transpose/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"   ?   ?   @   
t
*deconv2/conv_transpose/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
v
,deconv2/conv_transpose/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
v
,deconv2/conv_transpose/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
$deconv2/conv_transpose/strided_sliceStridedSlicedeconv2/conv_transpose/Shape*deconv2/conv_transpose/strided_slice/stack,deconv2/conv_transpose/strided_slice/stack_1,deconv2/conv_transpose/strided_slice/stack_2*
Index0*
shrink_axis_mask*
_output_shapes
: *

begin_mask *
end_mask *
ellipsis_mask *
T0*
new_axis_mask 
w
deconv2/conv_transpose/Shape_1Const*%
valueB"   ?   ?   @   *
dtype0*
_output_shapes
:
v
,deconv2/conv_transpose/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
x
.deconv2/conv_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
x
.deconv2/conv_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
&deconv2/conv_transpose/strided_slice_1StridedSlicedeconv2/conv_transpose/Shape_1,deconv2/conv_transpose/strided_slice_1/stack.deconv2/conv_transpose/strided_slice_1/stack_1.deconv2/conv_transpose/strided_slice_1/stack_2*
Index0*
ellipsis_mask *
new_axis_mask *

begin_mask *
_output_shapes
: *
end_mask *
shrink_axis_mask*
T0
^
deconv2/conv_transpose/mul/yConst*
_output_shapes
: *
value	B :*
dtype0
?
deconv2/conv_transpose/mulMul$deconv2/conv_transpose/strided_slicedeconv2/conv_transpose/mul/y*
T0*
_output_shapes
: 
`
deconv2/conv_transpose/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
deconv2/conv_transpose/mul_1Muldeconv2/conv_transpose/muldeconv2/conv_transpose/mul_1/y*
T0*
_output_shapes
: 
`
deconv2/conv_transpose/mul_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
deconv2/conv_transpose/mul_2Mul&deconv2/conv_transpose/strided_slice_1deconv2/conv_transpose/mul_2/y*
_output_shapes
: *
T0
`
deconv2/conv_transpose/mul_3/yConst*
dtype0*
value	B :*
_output_shapes
: 
?
deconv2/conv_transpose/mul_3Muldeconv2/conv_transpose/mul_2deconv2/conv_transpose/mul_3/y*
_output_shapes
: *
T0
?
"deconv2/conv_transpose/resize/sizePackdeconv2/conv_transpose/mul_1deconv2/conv_transpose/mul_3*
N*

axis *
T0*
_output_shapes
:
?
3deconv2/conv_transpose/resize/ResizeNearestNeighborResizeNearestNeighbordeconv1/Select"deconv2/conv_transpose/resize/size*(
_output_shapes
:??@*
align_corners( *
T0
?
2deconv2/conv_transpose/conv/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"      @       *
dtype0
v
1deconv2/conv_transpose/conv/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
x
3deconv2/conv_transpose/conv/truncated_normal/stddevConst*
valueB
 *???=*
_output_shapes
: *
dtype0
?
<deconv2/conv_transpose/conv/truncated_normal/TruncatedNormalTruncatedNormal2deconv2/conv_transpose/conv/truncated_normal/shape*
T0*
seed2 *
dtype0*&
_output_shapes
:@ *

seed 
?
0deconv2/conv_transpose/conv/truncated_normal/mulMul<deconv2/conv_transpose/conv/truncated_normal/TruncatedNormal3deconv2/conv_transpose/conv/truncated_normal/stddev*
T0*&
_output_shapes
:@ 
?
,deconv2/conv_transpose/conv/truncated_normalAdd0deconv2/conv_transpose/conv/truncated_normal/mul1deconv2/conv_transpose/conv/truncated_normal/mean*&
_output_shapes
:@ *
T0
?
"deconv2/conv_transpose/conv/weight
VariableV2*
dtype0*&
_output_shapes
:@ *
shared_name *
shape:@ *
	container 
?
)deconv2/conv_transpose/conv/weight/AssignAssign"deconv2/conv_transpose/conv/weight,deconv2/conv_transpose/conv/truncated_normal*&
_output_shapes
:@ *
use_locking(*5
_class+
)'loc:@deconv2/conv_transpose/conv/weight*
T0*
validate_shape(
?
'deconv2/conv_transpose/conv/weight/readIdentity"deconv2/conv_transpose/conv/weight*&
_output_shapes
:@ *
T0*5
_class+
)'loc:@deconv2/conv_transpose/conv/weight
?
.deconv2/conv_transpose/conv/MirrorPad/paddingsConst*9
value0B."                             *
_output_shapes

:*
dtype0
?
%deconv2/conv_transpose/conv/MirrorPad	MirrorPad3deconv2/conv_transpose/resize/ResizeNearestNeighbor.deconv2/conv_transpose/conv/MirrorPad/paddings*(
_output_shapes
:??@*
mode	REFLECT*
	Tpaddings0*
T0
?
 deconv2/conv_transpose/conv/convConv2D%deconv2/conv_transpose/conv/MirrorPad'deconv2/conv_transpose/conv/weight/read*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*(
_output_shapes
:?? *
	dilations

w
&deconv2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
?
deconv2/moments/meanMean deconv2/conv_transpose/conv/conv&deconv2/moments/mean/reduction_indices*
	keep_dims(*
T0*&
_output_shapes
: *

Tidx0
s
deconv2/moments/StopGradientStopGradientdeconv2/moments/mean*&
_output_shapes
: *
T0
?
!deconv2/moments/SquaredDifferenceSquaredDifference deconv2/conv_transpose/conv/convdeconv2/moments/StopGradient*(
_output_shapes
:?? *
T0
{
*deconv2/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB"      *
dtype0
?
deconv2/moments/varianceMean!deconv2/moments/SquaredDifference*deconv2/moments/variance/reduction_indices*

Tidx0*&
_output_shapes
: *
	keep_dims(*
T0
}
deconv2/SubSub deconv2/conv_transpose/conv/convdeconv2/moments/mean*
T0*(
_output_shapes
:?? 
R
deconv2/Add/yConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0
l
deconv2/AddAdddeconv2/moments/variancedeconv2/Add/y*
T0*&
_output_shapes
: 
R
deconv2/SqrtSqrtdeconv2/Add*&
_output_shapes
: *
T0
d
deconv2/divRealDivdeconv2/Subdeconv2/Sqrt*
T0*(
_output_shapes
:?? 
T
deconv2/ReluReludeconv2/div*
T0*(
_output_shapes
:?? 
e
deconv2/EqualEqualdeconv2/Reludeconv2/Relu*(
_output_shapes
:?? *
T0
{
"deconv2/zeros_like/shape_as_tensorConst*
_output_shapes
:*%
valueB"   ?  ?      *
dtype0
]
deconv2/zeros_like/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
?
deconv2/zeros_likeFill"deconv2/zeros_like/shape_as_tensordeconv2/zeros_like/Const*

index_type0*
T0*(
_output_shapes
:?? 
|
deconv2/SelectSelectdeconv2/Equaldeconv2/Reludeconv2/zeros_like*
T0*(
_output_shapes
:?? 
|
#deconv3/conv/truncated_normal/shapeConst*
dtype0*%
valueB"	   	          *
_output_shapes
:
g
"deconv3/conv/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
i
$deconv3/conv/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *???=
?
-deconv3/conv/truncated_normal/TruncatedNormalTruncatedNormal#deconv3/conv/truncated_normal/shape*
T0*
seed2 *

seed *&
_output_shapes
:		 *
dtype0
?
!deconv3/conv/truncated_normal/mulMul-deconv3/conv/truncated_normal/TruncatedNormal$deconv3/conv/truncated_normal/stddev*
T0*&
_output_shapes
:		 
?
deconv3/conv/truncated_normalAdd!deconv3/conv/truncated_normal/mul"deconv3/conv/truncated_normal/mean*&
_output_shapes
:		 *
T0
?
deconv3/conv/weight
VariableV2*
shape:		 *
	container *&
_output_shapes
:		 *
shared_name *
dtype0
?
deconv3/conv/weight/AssignAssigndeconv3/conv/weightdeconv3/conv/truncated_normal*
T0*
use_locking(*&
_output_shapes
:		 *
validate_shape(*&
_class
loc:@deconv3/conv/weight
?
deconv3/conv/weight/readIdentitydeconv3/conv/weight*&
_class
loc:@deconv3/conv/weight*&
_output_shapes
:		 *
T0
?
deconv3/conv/MirrorPad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
?
deconv3/conv/MirrorPad	MirrorPaddeconv2/Selectdeconv3/conv/MirrorPad/paddings*(
_output_shapes
:?? *
T0*
	Tpaddings0*
mode	REFLECT
?
deconv3/conv/convConv2Ddeconv3/conv/MirrorPaddeconv3/conv/weight/read*
	dilations
*(
_output_shapes
:??*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC
w
&deconv3/moments/mean/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
?
deconv3/moments/meanMeandeconv3/conv/conv&deconv3/moments/mean/reduction_indices*&
_output_shapes
:*
T0*
	keep_dims(*

Tidx0
s
deconv3/moments/StopGradientStopGradientdeconv3/moments/mean*&
_output_shapes
:*
T0
?
!deconv3/moments/SquaredDifferenceSquaredDifferencedeconv3/conv/convdeconv3/moments/StopGradient*(
_output_shapes
:??*
T0
{
*deconv3/moments/variance/reduction_indicesConst*
dtype0*
valueB"      *
_output_shapes
:
?
deconv3/moments/varianceMean!deconv3/moments/SquaredDifference*deconv3/moments/variance/reduction_indices*

Tidx0*
T0*&
_output_shapes
:*
	keep_dims(
n
deconv3/SubSubdeconv3/conv/convdeconv3/moments/mean*(
_output_shapes
:??*
T0
R
deconv3/Add/yConst*
dtype0*
_output_shapes
: *
valueB
 *_p?0
l
deconv3/AddAdddeconv3/moments/variancedeconv3/Add/y*&
_output_shapes
:*
T0
R
deconv3/SqrtSqrtdeconv3/Add*
T0*&
_output_shapes
:
d
deconv3/divRealDivdeconv3/Subdeconv3/Sqrt*
T0*(
_output_shapes
:??
T
deconv3/TanhTanhdeconv3/div*(
_output_shapes
:??*
T0
J
add/yConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
R
addAdddeconv3/Tanhadd/y*
T0*(
_output_shapes
:??
J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B
I
mulMuladdmul/y*
T0*(
_output_shapes
:??
^
ShapeConst*%
valueB"   ?  ?     *
dtype0*
_output_shapes
:
]
strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
_
strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
end_mask *
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
T0*
_output_shapes
: *
new_axis_mask 
`
Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"   ?  ?     
_
strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
strided_slice_1StridedSliceShape_1strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
Index0*

begin_mask *
end_mask *
_output_shapes
: *
shrink_axis_mask*
new_axis_mask *
ellipsis_mask *
T0
G
sub/yConst*
value	B :*
_output_shapes
: *
dtype0
A
subSubstrided_slicesub/y*
T0*
_output_shapes
: 
I
sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :
G
sub_1Substrided_slice_1sub_1/y*
_output_shapes
: *
T0
R
stack/0Const*
_output_shapes
: *
valueB :
?????????*
dtype0
R
stack/3Const*
dtype0*
valueB :
?????????*
_output_shapes
: 
e
stackPackstack/0subsub_1stack/3*
T0*

axis *
_output_shapes
:*
N
e
output/beginConst*
_output_shapes
:*
dtype0*%
valueB"    
   
       
i
outputSlicemuloutput/beginstack*
T0*
Index0*(
_output_shapes
:??
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
?
save/save/tensor_namesConst*?
value?B?Bconv1/conv/weightBconv2/conv/weightBconv3/conv/weightB"deconv1/conv_transpose/conv/weightB"deconv2/conv_transpose/conv/weightBdeconv3/conv/weightBres1/residual/conv/weightBres1/residual/conv_1/weightBres2/residual/conv/weightBres2/residual/conv_1/weightBres3/residual/conv/weightBres3/residual/conv_1/weightBres4/residual/conv/weightBres4/residual/conv_1/weightBres5/residual/conv/weightBres5/residual/conv_1/weight*
dtype0*
_output_shapes
:
?
save/save/shapes_and_slicesConst*
dtype0*
_output_shapes
:*3
value*B(B B B B B B B B B B B B B B B B 
?
	save/save
SaveSlices
save/Constsave/save/tensor_namessave/save/shapes_and_slicesconv1/conv/weightconv2/conv/weightconv3/conv/weight"deconv1/conv_transpose/conv/weight"deconv2/conv_transpose/conv/weightdeconv3/conv/weightres1/residual/conv/weightres1/residual/conv_1/weightres2/residual/conv/weightres2/residual/conv_1/weightres3/residual/conv/weightres3/residual/conv_1/weightres4/residual/conv/weightres4/residual/conv_1/weightres5/residual/conv/weightres5/residual/conv_1/weight*
T
2
{
save/control_dependencyIdentity
save/Const
^save/save*
_output_shapes
: *
T0*
_class
loc:@save/Const
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*?
value?B?Bconv1/conv/weightBconv2/conv/weightBconv3/conv/weightB"deconv1/conv_transpose/conv/weightB"deconv2/conv_transpose/conv/weightBdeconv3/conv/weightBres1/residual/conv/weightBres1/residual/conv_1/weightBres2/residual/conv/weightBres2/residual/conv_1/weightBres3/residual/conv/weightBres3/residual/conv_1/weightBres4/residual/conv/weightBres4/residual/conv_1/weightBres5/residual/conv/weightBres5/residual/conv_1/weight*
_output_shapes
:
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2*T
_output_shapesB
@::::::::::::::::
?
save/AssignAssignconv1/conv/weightsave/RestoreV2*&
_output_shapes
:		 *
use_locking(*$
_class
loc:@conv1/conv/weight*
T0*
validate_shape(
?
save/Assign_1Assignconv2/conv/weightsave/RestoreV2:1*&
_output_shapes
: @*
use_locking(*
T0*$
_class
loc:@conv2/conv/weight*
validate_shape(
?
save/Assign_2Assignconv3/conv/weightsave/RestoreV2:2*
use_locking(*$
_class
loc:@conv3/conv/weight*
validate_shape(*'
_output_shapes
:@?*
T0
?
save/Assign_3Assign"deconv1/conv_transpose/conv/weightsave/RestoreV2:3*
use_locking(*'
_output_shapes
:?@*
validate_shape(*5
_class+
)'loc:@deconv1/conv_transpose/conv/weight*
T0
?
save/Assign_4Assign"deconv2/conv_transpose/conv/weightsave/RestoreV2:4*
use_locking(*&
_output_shapes
:@ *
validate_shape(*
T0*5
_class+
)'loc:@deconv2/conv_transpose/conv/weight
?
save/Assign_5Assigndeconv3/conv/weightsave/RestoreV2:5*
use_locking(*
validate_shape(*
T0*&
_class
loc:@deconv3/conv/weight*&
_output_shapes
:		 
?
save/Assign_6Assignres1/residual/conv/weightsave/RestoreV2:6*(
_output_shapes
:??*
validate_shape(*
T0*
use_locking(*,
_class"
 loc:@res1/residual/conv/weight
?
save/Assign_7Assignres1/residual/conv_1/weightsave/RestoreV2:7*
T0*.
_class$
" loc:@res1/residual/conv_1/weight*
use_locking(*
validate_shape(*(
_output_shapes
:??
?
save/Assign_8Assignres2/residual/conv/weightsave/RestoreV2:8*
use_locking(*
validate_shape(*(
_output_shapes
:??*,
_class"
 loc:@res2/residual/conv/weight*
T0
?
save/Assign_9Assignres2/residual/conv_1/weightsave/RestoreV2:9*
validate_shape(*.
_class$
" loc:@res2/residual/conv_1/weight*
use_locking(*
T0*(
_output_shapes
:??
?
save/Assign_10Assignres3/residual/conv/weightsave/RestoreV2:10*(
_output_shapes
:??*
validate_shape(*
T0*
use_locking(*,
_class"
 loc:@res3/residual/conv/weight
?
save/Assign_11Assignres3/residual/conv_1/weightsave/RestoreV2:11*.
_class$
" loc:@res3/residual/conv_1/weight*
validate_shape(*(
_output_shapes
:??*
T0*
use_locking(
?
save/Assign_12Assignres4/residual/conv/weightsave/RestoreV2:12*
T0*(
_output_shapes
:??*,
_class"
 loc:@res4/residual/conv/weight*
validate_shape(*
use_locking(
?
save/Assign_13Assignres4/residual/conv_1/weightsave/RestoreV2:13*
use_locking(*.
_class$
" loc:@res4/residual/conv_1/weight*(
_output_shapes
:??*
validate_shape(*
T0
?
save/Assign_14Assignres5/residual/conv/weightsave/RestoreV2:14*
T0*
use_locking(*
validate_shape(*,
_class"
 loc:@res5/residual/conv/weight*(
_output_shapes
:??
?
save/Assign_15Assignres5/residual/conv_1/weightsave/RestoreV2:15*.
_class$
" loc:@res5/residual/conv_1/weight*
validate_shape(*
T0*(
_output_shapes
:??*
use_locking(
?
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
?
initNoOp^conv1/conv/weight/Assign^conv2/conv/weight/Assign^conv3/conv/weight/Assign*^deconv1/conv_transpose/conv/weight/Assign*^deconv2/conv_transpose/conv/weight/Assign^deconv3/conv/weight/Assign!^res1/residual/conv/weight/Assign#^res1/residual/conv_1/weight/Assign!^res2/residual/conv/weight/Assign#^res2/residual/conv_1/weight/Assign!^res3/residual/conv/weight/Assign#^res3/residual/conv_1/weight/Assign!^res4/residual/conv/weight/Assign#^res4/residual/conv_1/weight/Assign!^res5/residual/conv/weight/Assign#^res5/residual/conv_1/weight/Assign

init_1NoOp
[
save_1/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
_output_shapes
: *
dtype0*
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
dtype0*
shape: *
_output_shapes
: 
?
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_ef750736a30c4b9d9c6584cb955483be/part*
_output_shapes
: *
dtype0
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 
?
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
?
save_1/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*?
value?B?Bconv1/conv/weightBconv2/conv/weightBconv3/conv/weightB"deconv1/conv_transpose/conv/weightB"deconv2/conv_transpose/conv/weightBdeconv3/conv/weightBres1/residual/conv/weightBres1/residual/conv_1/weightBres2/residual/conv/weightBres2/residual/conv_1/weightBres3/residual/conv/weightBres3/residual/conv_1/weightBres4/residual/conv/weightBres4/residual/conv_1/weightBres5/residual/conv/weightBres5/residual/conv_1/weight*
dtype0
?
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
?
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesconv1/conv/weightconv2/conv/weightconv3/conv/weight"deconv1/conv_transpose/conv/weight"deconv2/conv_transpose/conv/weightdeconv3/conv/weightres1/residual/conv/weightres1/residual/conv_1/weightres2/residual/conv/weightres2/residual/conv_1/weightres3/residual/conv/weightres3/residual/conv_1/weightres4/residual/conv/weightres4/residual/conv_1/weightres5/residual/conv/weightres5/residual/conv_1/weight"/device:CPU:0*
dtypes
2
?
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
?
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
_output_shapes
:*

axis *
T0*
N
?
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(
?
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?Bconv1/conv/weightBconv2/conv/weightBconv3/conv/weightB"deconv1/conv_transpose/conv/weightB"deconv2/conv_transpose/conv/weightBdeconv3/conv/weightBres1/residual/conv/weightBres1/residual/conv_1/weightBres2/residual/conv/weightBres2/residual/conv_1/weightBres3/residual/conv/weightBres3/residual/conv_1/weightBres4/residual/conv/weightBres4/residual/conv_1/weightBres5/residual/conv/weightBres5/residual/conv_1/weight*
_output_shapes
:*
dtype0
?
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*3
value*B(B B B B B B B B B B B B B B B B *
dtype0
?
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2*T
_output_shapesB
@::::::::::::::::
?
save_1/AssignAssignconv1/conv/weightsave_1/RestoreV2*
validate_shape(*
T0*$
_class
loc:@conv1/conv/weight*&
_output_shapes
:		 *
use_locking(
?
save_1/Assign_1Assignconv2/conv/weightsave_1/RestoreV2:1*
use_locking(*
validate_shape(*&
_output_shapes
: @*$
_class
loc:@conv2/conv/weight*
T0
?
save_1/Assign_2Assignconv3/conv/weightsave_1/RestoreV2:2*'
_output_shapes
:@?*$
_class
loc:@conv3/conv/weight*
T0*
validate_shape(*
use_locking(
?
save_1/Assign_3Assign"deconv1/conv_transpose/conv/weightsave_1/RestoreV2:3*
validate_shape(*
use_locking(*5
_class+
)'loc:@deconv1/conv_transpose/conv/weight*'
_output_shapes
:?@*
T0
?
save_1/Assign_4Assign"deconv2/conv_transpose/conv/weightsave_1/RestoreV2:4*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@deconv2/conv_transpose/conv/weight*&
_output_shapes
:@ 
?
save_1/Assign_5Assigndeconv3/conv/weightsave_1/RestoreV2:5*&
_class
loc:@deconv3/conv/weight*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:		 
?
save_1/Assign_6Assignres1/residual/conv/weightsave_1/RestoreV2:6*
validate_shape(*
T0*(
_output_shapes
:??*
use_locking(*,
_class"
 loc:@res1/residual/conv/weight
?
save_1/Assign_7Assignres1/residual/conv_1/weightsave_1/RestoreV2:7*(
_output_shapes
:??*
validate_shape(*
use_locking(*
T0*.
_class$
" loc:@res1/residual/conv_1/weight
?
save_1/Assign_8Assignres2/residual/conv/weightsave_1/RestoreV2:8*
use_locking(*(
_output_shapes
:??*,
_class"
 loc:@res2/residual/conv/weight*
T0*
validate_shape(
?
save_1/Assign_9Assignres2/residual/conv_1/weightsave_1/RestoreV2:9*.
_class$
" loc:@res2/residual/conv_1/weight*
T0*
validate_shape(*(
_output_shapes
:??*
use_locking(
?
save_1/Assign_10Assignres3/residual/conv/weightsave_1/RestoreV2:10*(
_output_shapes
:??*
T0*
validate_shape(*,
_class"
 loc:@res3/residual/conv/weight*
use_locking(
?
save_1/Assign_11Assignres3/residual/conv_1/weightsave_1/RestoreV2:11*(
_output_shapes
:??*
T0*.
_class$
" loc:@res3/residual/conv_1/weight*
validate_shape(*
use_locking(
?
save_1/Assign_12Assignres4/residual/conv/weightsave_1/RestoreV2:12*
use_locking(*
T0*,
_class"
 loc:@res4/residual/conv/weight*(
_output_shapes
:??*
validate_shape(
?
save_1/Assign_13Assignres4/residual/conv_1/weightsave_1/RestoreV2:13*
validate_shape(*.
_class$
" loc:@res4/residual/conv_1/weight*
use_locking(*(
_output_shapes
:??*
T0
?
save_1/Assign_14Assignres5/residual/conv/weightsave_1/RestoreV2:14*
T0*,
_class"
 loc:@res5/residual/conv/weight*(
_output_shapes
:??*
validate_shape(*
use_locking(
?
save_1/Assign_15Assignres5/residual/conv_1/weightsave_1/RestoreV2:15*.
_class$
" loc:@res5/residual/conv_1/weight*
T0*
use_locking(*
validate_shape(*(
_output_shapes
:??
?
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"?
	variables??
j
conv1/conv/weight:0conv1/conv/weight/Assignconv1/conv/weight/read:02conv1/conv/truncated_normal:08
j
conv2/conv/weight:0conv2/conv/weight/Assignconv2/conv/weight/read:02conv2/conv/truncated_normal:08
j
conv3/conv/weight:0conv3/conv/weight/Assignconv3/conv/weight/read:02conv3/conv/truncated_normal:08
?
res1/residual/conv/weight:0 res1/residual/conv/weight/Assign res1/residual/conv/weight/read:02%res1/residual/conv/truncated_normal:08
?
res1/residual/conv_1/weight:0"res1/residual/conv_1/weight/Assign"res1/residual/conv_1/weight/read:02'res1/residual/conv_1/truncated_normal:08
?
res2/residual/conv/weight:0 res2/residual/conv/weight/Assign res2/residual/conv/weight/read:02%res2/residual/conv/truncated_normal:08
?
res2/residual/conv_1/weight:0"res2/residual/conv_1/weight/Assign"res2/residual/conv_1/weight/read:02'res2/residual/conv_1/truncated_normal:08
?
res3/residual/conv/weight:0 res3/residual/conv/weight/Assign res3/residual/conv/weight/read:02%res3/residual/conv/truncated_normal:08
?
res3/residual/conv_1/weight:0"res3/residual/conv_1/weight/Assign"res3/residual/conv_1/weight/read:02'res3/residual/conv_1/truncated_normal:08
?
res4/residual/conv/weight:0 res4/residual/conv/weight/Assign res4/residual/conv/weight/read:02%res4/residual/conv/truncated_normal:08
?
res4/residual/conv_1/weight:0"res4/residual/conv_1/weight/Assign"res4/residual/conv_1/weight/read:02'res4/residual/conv_1/truncated_normal:08
?
res5/residual/conv/weight:0 res5/residual/conv/weight/Assign res5/residual/conv/weight/read:02%res5/residual/conv/truncated_normal:08
?
res5/residual/conv_1/weight:0"res5/residual/conv_1/weight/Assign"res5/residual/conv_1/weight/read:02'res5/residual/conv_1/truncated_normal:08
?
$deconv1/conv_transpose/conv/weight:0)deconv1/conv_transpose/conv/weight/Assign)deconv1/conv_transpose/conv/weight/read:02.deconv1/conv_transpose/conv/truncated_normal:08
?
$deconv2/conv_transpose/conv/weight:0)deconv2/conv_transpose/conv/weight/Assign)deconv2/conv_transpose/conv/weight/read:02.deconv2/conv_transpose/conv/truncated_normal:08
r
deconv3/conv/weight:0deconv3/conv/weight/Assigndeconv3/conv/weight/read:02deconv3/conv/truncated_normal:08"?
trainable_variables??
j
conv1/conv/weight:0conv1/conv/weight/Assignconv1/conv/weight/read:02conv1/conv/truncated_normal:08
j
conv2/conv/weight:0conv2/conv/weight/Assignconv2/conv/weight/read:02conv2/conv/truncated_normal:08
j
conv3/conv/weight:0conv3/conv/weight/Assignconv3/conv/weight/read:02conv3/conv/truncated_normal:08
?
res1/residual/conv/weight:0 res1/residual/conv/weight/Assign res1/residual/conv/weight/read:02%res1/residual/conv/truncated_normal:08
?
res1/residual/conv_1/weight:0"res1/residual/conv_1/weight/Assign"res1/residual/conv_1/weight/read:02'res1/residual/conv_1/truncated_normal:08
?
res2/residual/conv/weight:0 res2/residual/conv/weight/Assign res2/residual/conv/weight/read:02%res2/residual/conv/truncated_normal:08
?
res2/residual/conv_1/weight:0"res2/residual/conv_1/weight/Assign"res2/residual/conv_1/weight/read:02'res2/residual/conv_1/truncated_normal:08
?
res3/residual/conv/weight:0 res3/residual/conv/weight/Assign res3/residual/conv/weight/read:02%res3/residual/conv/truncated_normal:08
?
res3/residual/conv_1/weight:0"res3/residual/conv_1/weight/Assign"res3/residual/conv_1/weight/read:02'res3/residual/conv_1/truncated_normal:08
?
res4/residual/conv/weight:0 res4/residual/conv/weight/Assign res4/residual/conv/weight/read:02%res4/residual/conv/truncated_normal:08
?
res4/residual/conv_1/weight:0"res4/residual/conv_1/weight/Assign"res4/residual/conv_1/weight/read:02'res4/residual/conv_1/truncated_normal:08
?
res5/residual/conv/weight:0 res5/residual/conv/weight/Assign res5/residual/conv/weight/read:02%res5/residual/conv/truncated_normal:08
?
res5/residual/conv_1/weight:0"res5/residual/conv_1/weight/Assign"res5/residual/conv_1/weight/read:02'res5/residual/conv_1/truncated_normal:08
?
$deconv1/conv_transpose/conv/weight:0)deconv1/conv_transpose/conv/weight/Assign)deconv1/conv_transpose/conv/weight/read:02.deconv1/conv_transpose/conv/truncated_normal:08
?
$deconv2/conv_transpose/conv/weight:0)deconv2/conv_transpose/conv/weight/Assign)deconv2/conv_transpose/conv/weight/read:02.deconv2/conv_transpose/conv/truncated_normal:08
r
deconv3/conv/weight:0deconv3/conv/weight/Assigndeconv3/conv/weight/read:02deconv3/conv/truncated_normal:08*?
serving_defaultr
(
input
input:0??*
output 
output:0??tensorflow/serving/predict