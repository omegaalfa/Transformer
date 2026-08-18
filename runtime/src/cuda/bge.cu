#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>

namespace {
constexpr int D = 384, I = 1536, H = 12, HD = 32, LAYERS = 12, PARAMS = 197;
struct Model {
    float* p[PARAMS]{}; size_t n[PARAMS]{}; cublasHandle_t blas{}; cudaStream_t stream{}; bool ready=false;
    int64_t *di=nullptr,*dt=nullptr; uint8_t* dm=nullptr;
    float *a=nullptr,*b=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,*merged=nullptr,*tmp=nullptr,*inter=nullptr,*result=nullptr;
    size_t row_capacity=0, batch_capacity=0;
    cudaGraph_t graph{}; cudaGraphExec_t graph_exec{}; int graph_batch=0,graph_seq=0,math_mode=0; bool graph_enabled=true,graph_warmed=false;
    uint64_t forward_mallocs=0,forward_frees=0,kernel_launches=0,parameter_uploads=0,workspace_reallocations=0,h2d_bytes=0,d2h_bytes=0,synchronizations=0;
    uint64_t graph_launches=0,internal_submissions=0,graph_captures=0,graph_invalidations=0;
    int last_batch=0,last_seq=0;bool last_graph_captured=false,last_graph_reused=false;
};

bool ok(cudaError_t e) { return e == cudaSuccess; }
bool okb(cublasStatus_t e) { return e == CUBLAS_STATUS_SUCCESS; }

void invalidate_graph(Model* m){
    bool existed=m->graph_exec||m->graph;if(m->graph_exec)cudaGraphExecDestroy(m->graph_exec);if(m->graph)cudaGraphDestroy(m->graph);
    if(existed)m->graph_invalidations++;m->graph_exec=nullptr;m->graph=nullptr;m->graph_batch=0;m->graph_seq=0;m->graph_warmed=false;
}
void release_workspace(Model* m){
    invalidate_graph(m);void* allocations[]={m->di,m->dt,m->dm,m->a,m->b,m->q,m->k,m->v,m->merged,m->tmp,m->inter,m->result};for(void* allocation:allocations)if(allocation)m->forward_frees++;
    cudaFree(m->di);cudaFree(m->dt);cudaFree(m->dm);cudaFree(m->a);cudaFree(m->b);cudaFree(m->q);cudaFree(m->k);cudaFree(m->v);cudaFree(m->merged);cudaFree(m->tmp);cudaFree(m->inter);cudaFree(m->result);
    m->di=m->dt=nullptr;m->dm=nullptr;m->a=m->b=m->q=m->k=m->v=m->merged=m->tmp=m->inter=m->result=nullptr;
    m->row_capacity=m->batch_capacity=0;
}
bool ensure_workspace(Model* m,int batch,int seq){
    if(batch<1||seq<1)return false;size_t rows=(size_t)batch*(size_t)seq;
    if(rows>std::numeric_limits<size_t>::max()/D||rows>std::numeric_limits<size_t>::max()/I)return false;
    size_t attention_rows=(size_t)batch*H*(size_t)seq;
    if(attention_rows>std::numeric_limits<size_t>::max()/(size_t)seq)return false;
    size_t probabilities=attention_rows*(size_t)seq;(void)probabilities;
    if(rows<=m->row_capacity&&(size_t)batch<=m->batch_capacity)return true;
    size_t new_rows=rows>m->row_capacity?rows:m->row_capacity;
    size_t new_batch=(size_t)batch>m->batch_capacity?(size_t)batch:m->batch_capacity;
    release_workspace(m);
    m->workspace_reallocations++;
    #define ALLOC_MODEL(ptr,bytes) if(!ok(cudaMalloc(&(m->ptr),(bytes)))){release_workspace(m);return false;}else m->forward_mallocs++
    ALLOC_MODEL(di,new_rows*sizeof(int64_t));ALLOC_MODEL(dt,new_rows*sizeof(int64_t));ALLOC_MODEL(dm,new_rows);
    ALLOC_MODEL(a,new_rows*D*sizeof(float));ALLOC_MODEL(b,new_rows*D*sizeof(float));ALLOC_MODEL(q,new_rows*D*sizeof(float));ALLOC_MODEL(k,new_rows*D*sizeof(float));ALLOC_MODEL(v,new_rows*D*sizeof(float));
    ALLOC_MODEL(merged,new_rows*D*sizeof(float));ALLOC_MODEL(tmp,new_rows*D*sizeof(float));ALLOC_MODEL(inter,new_rows*I*sizeof(float));ALLOC_MODEL(result,new_batch*D*sizeof(float));
    m->row_capacity=new_rows;m->batch_capacity=new_batch;return true;
}

__global__ void embeddings(const int64_t* ids, const int64_t* types, float* out,
    const float* word, const float* pos, const float* type, int rows, int seq) {
    int index=blockIdx.x*blockDim.x+threadIdx.x, total=rows*seq*D;
    if(index<total){ int token=index/D, d=index%D; out[index]=word[ids[token]*D+d]+pos[(token%seq)*D+d]+type[types[token]*D+d]; }
}
__global__ void layer_norm(const float* a,const float* residual,float* out,const float* gamma,const float* beta,int rows){
    __shared__ double means[128],m2s[128];__shared__ int counts[128];
    int r=blockIdx.x,tid=threadIdx.x;if(r>=rows)return;double mean=0,m2=0;int count=0;
    for(int d=tid;d<D;d+=blockDim.x){double x=(double)a[r*D+d]+(residual?(double)residual[r*D+d]:0.0);count++;double delta=x-mean;mean+=delta/count;m2+=delta*(x-mean);}
    means[tid]=mean;m2s[tid]=m2;counts[tid]=count;__syncthreads();
    for(int offset=64;offset>0;offset/=2){if(tid<offset){int right=tid+offset,n1=counts[tid],n2=counts[right];if(n2){double delta=means[right]-means[tid],n=n1+n2;means[tid]+=delta*n2/n;m2s[tid]+=m2s[right]+delta*delta*n1*n2/n;counts[tid]=n;}}__syncthreads();}
    double inv=rsqrt(m2s[0]/D+1e-12);for(int d=tid;d<D;d+=blockDim.x){double x=(double)a[r*D+d]+(residual?(double)residual[r*D+d]:0.0);out[r*D+d]=gamma[d]*(float)((x-means[0])*inv)+beta[d];}
}
__global__ void bias(float* out,const float* b,int rows,int width){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*width)out[i]+=b[i%width];}
__global__ void gelu(float* x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];x[i]=0.5f*v*(1.0f+erff(v*0.7071067811865475f));}}
__global__ void fused_attention(const float* q,const float* k,const float* v,const uint8_t* mask,float* merged,int batch,int seq){
    __shared__ float probabilities[512];__shared__ float reduction[128];int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;if(row>=batch*H*seq)return;int query=row%seq,head=(row/seq)%H,b=row/(H*seq);
    for(int key=warp;key<seq;key+=4){float score=q[(b*seq+query)*D+head*HD+lane]*k[(b*seq+key)*D+head*HD+lane];for(int offset=16;offset>0;offset/=2)score+=__shfl_down_sync(0xffffffff,score,offset);if(lane==0)probabilities[key]=mask[b*seq+key]?score*0.1767766952966369f:-INFINITY;}
    __syncthreads();float local_max=-INFINITY;for(int key=tid;key<seq;key+=blockDim.x)local_max=fmaxf(local_max,probabilities[key]);reduction[tid]=local_max;__syncthreads();for(int offset=64;offset>0;offset/=2){if(tid<offset)reduction[tid]=fmaxf(reduction[tid],reduction[tid+offset]);__syncthreads();}
    float local_sum=0;for(int key=tid;key<seq;key+=blockDim.x){float value=mask[b*seq+key]?expf(probabilities[key]-reduction[0]):0;probabilities[key]=value;local_sum+=value;}reduction[tid]=local_sum;__syncthreads();for(int offset=64;offset>0;offset/=2){if(tid<offset)reduction[tid]+=reduction[tid+offset];__syncthreads();}for(int key=tid;key<seq;key+=blockDim.x)probabilities[key]/=reduction[0];__syncthreads();
    if(tid<HD){float sum=0;int d=head*HD+tid;for(int key=0;key<seq;key++)sum+=probabilities[key]*v[(b*seq+key)*D+d];merged[(b*seq+query)*D+d]=sum;}
}
__global__ void cls_l2(const float* hidden,float* out,int batch,int seq){__shared__ double sums[128];int b=blockIdx.x,tid=threadIdx.x;if(b>=batch)return;double sum=0;for(int d=tid;d<D;d+=blockDim.x){double value=hidden[b*seq*D+d];sum+=value*value;}sums[tid]=sum;__syncthreads();for(int offset=64;offset>0;offset/=2){if(tid<offset)sums[tid]+=sums[tid+offset];__syncthreads();}double inv=rsqrt(sums[0]);for(int d=tid;d<D;d+=blockDim.x)out[b*D+d]=(float)(hidden[b*seq*D+d]*inv);}


bool linear(Model* m,const float* a,const float* w,const float* b,float* c,int rows,int k,int n){
    const float alpha=1,beta=0; if(!okb(cublasSgemm(m->blas,CUBLAS_OP_N,CUBLAS_OP_N,n,rows,k,&alpha,w,n,a,k,&beta,c,n)))return false;
    bias<<<(rows*n+255)/256,256,0,m->stream>>>(c,b,rows,n); return ok(cudaGetLastError());
}
bool linear_gelu(Model* m,const float* a,const float* w,const float* b,float* c,int rows,int k,int n){
    const float alpha=1,beta=0;if(!okb(cublasSgemm(m->blas,CUBLAS_OP_N,CUBLAS_OP_N,n,rows,k,&alpha,w,n,a,k,&beta,c,n)))return false;
    bias<<<(rows*n+255)/256,256,0,m->stream>>>(c,b,rows,n);gelu<<<(rows*n+255)/256,256,0,m->stream>>>(c,rows*n);return ok(cudaGetLastError());
}
bool launch_pipeline(Model* m,int batch,int seq,int rows,size_t dn){
    embeddings<<<(dn+255)/256,256,0,m->stream>>>(m->di,m->dt,m->tmp,m->p[0],m->p[1],m->p[2],batch,seq);layer_norm<<<rows,128,0,m->stream>>>(m->tmp,nullptr,m->a,m->p[3],m->p[4],rows);
    for(int layer=0;layer<LAYERS;layer++){int x=5+layer*16;
        if(!linear(m,m->a,m->p[x],m->p[x+1],m->q,rows,D,D)||!linear(m,m->a,m->p[x+2],m->p[x+3],m->k,rows,D,D)||!linear(m,m->a,m->p[x+4],m->p[x+5],m->v,rows,D,D))return false;
        fused_attention<<<batch*H*seq,128,0,m->stream>>>(m->q,m->k,m->v,m->dm,m->merged,batch,seq);if(!linear(m,m->merged,m->p[x+6],m->p[x+7],m->tmp,rows,D,D))return false;
        layer_norm<<<rows,128,0,m->stream>>>(m->tmp,m->a,m->b,m->p[x+8],m->p[x+9],rows);if(!linear_gelu(m,m->b,m->p[x+10],m->p[x+11],m->inter,rows,D,I))return false;if(!linear(m,m->inter,m->p[x+12],m->p[x+13],m->tmp,rows,I,D))return false;layer_norm<<<rows,128,0,m->stream>>>(m->tmp,m->b,m->a,m->p[x+14],m->p[x+15],rows);
    }
    cls_l2<<<batch,128,0,m->stream>>>(m->a,m->result,batch,seq);return ok(cudaGetLastError());
}
int expected_count(int index){
    if(index==0)return 30522*D;if(index==1)return 512*D;if(index==2)return 2*D;if(index<5)return D;
    int slot=(index-5)%16; if(slot==0||slot==2||slot==4||slot==6)return D*D;
    if(slot==10)return D*I;if(slot==11)return I;if(slot==12)return I*D;
    return D;
}
}

extern "C" int cuda_bge_available_impl(){int count=0;return ok(cudaGetDeviceCount(&count))&&count>0;}
extern "C" void* cuda_bge_create_impl(){Model* m=new(std::nothrow) Model;if(!m)return nullptr;if(!ok(cudaStreamCreateWithFlags(&m->stream,cudaStreamNonBlocking))||!okb(cublasCreate(&m->blas))||!okb(cublasSetStream(m->blas,m->stream))||!okb(cublasSetMathMode(m->blas,CUBLAS_PEDANTIC_MATH))){if(m->blas)cublasDestroy(m->blas);if(m->stream)cudaStreamDestroy(m->stream);delete m;return nullptr;}return m;}
extern "C" int cuda_bge_set_parameter_impl(void* handle,int index,const float* data,size_t count){
    Model* m=(Model*)handle;if(!m||!data||index<0||index>=PARAMS||m->p[index]||count!=(size_t)expected_count(index))return 1;
    if(count>std::numeric_limits<size_t>::max()/sizeof(float)||!ok(cudaMalloc(&m->p[index],count*sizeof(float))))return 2;
    if(!ok(cudaMemcpy(m->p[index],data,count*sizeof(float),cudaMemcpyHostToDevice))){cudaFree(m->p[index]);m->p[index]=nullptr;m->n[index]=0;return 2;}m->n[index]=count;m->parameter_uploads++;return 0;
}
extern "C" int cuda_bge_finalize_impl(void* handle){Model*m=(Model*)handle;if(!m)return 1;for(int i=0;i<PARAMS;i++)if(!m->p[i])return 1;m->ready=true;return 0;}
extern "C" int cuda_bge_set_math_mode_impl(void* handle,int mode){Model*m=(Model*)handle;if(!m||(mode!=0&&mode!=1))return 1;cublasMath_t selected=mode==0?CUBLAS_PEDANTIC_MATH:CUBLAS_TF32_TENSOR_OP_MATH;if(!okb(cublasSetMathMode(m->blas,selected)))return 1;if(m->math_mode!=mode)invalidate_graph(m);m->math_mode=mode;return 0;}
extern "C" int cuda_bge_set_graph_enabled_impl(void* handle,int enabled){Model*m=(Model*)handle;if(!m||(enabled!=0&&enabled!=1))return 1;if(m->graph_enabled!=(enabled!=0))invalidate_graph(m);m->graph_enabled=enabled!=0;return 0;}
int cuda_bge_forward_core(void* handle,const int64_t* ids,const uint8_t* mask,const int64_t* types,int batch,int seq,float* host_out,float* timings,int timing_capacity,int* timing_count,float* phase_timings){
    Model*m=(Model*)handle;if(!m||!m->ready||!ids||!mask||!types||!host_out||batch<1||seq<1||seq>512)return 1;
    if((size_t)batch>std::numeric_limits<size_t>::max()/(size_t)seq)return 1;size_t row_count=(size_t)batch*(size_t)seq;
    if(row_count>(size_t)std::numeric_limits<int>::max()||row_count>std::numeric_limits<size_t>::max()/D)return 1;int rows=(int)row_count;size_t dn=row_count*D;
    for(int index=0;index<rows;index++)if(ids[index]<0||ids[index]>=30522||types[index]<0||types[index]>=2)return 1;
    m->forward_mallocs=0;m->forward_frees=0;m->kernel_launches=195;m->internal_submissions=195;m->h2d_bytes=row_count*(sizeof(int64_t)*2+sizeof(uint8_t));m->d2h_bytes=(size_t)batch*D*sizeof(float);m->synchronizations=1;m->last_batch=batch;m->last_seq=seq;m->last_graph_captured=false;m->last_graph_reused=false;
    if(!ensure_workspace(m,batch,seq))return 2;
    cudaEvent_t events[138]{};int marks=0;if(timings){if(timing_capacity<137||!timing_count)return 1;for(int i=0;i<138;i++)if(!ok(cudaEventCreate(&events[i])))return 2;cudaEventRecord(events[marks++],m->stream);}
    cudaEvent_t phases[4]{};if(phase_timings){for(int i=0;i<4;i++)if(!ok(cudaEventCreate(&phases[i])))return 2;cudaEventRecord(phases[0],m->stream);}
    #define MARK_STAGE() if(timings)cudaEventRecord(events[marks++],m->stream)
    if(!ok(cudaMemcpyAsync(m->di,ids,rows*sizeof(int64_t),cudaMemcpyHostToDevice,m->stream))||!ok(cudaMemcpyAsync(m->dt,types,rows*sizeof(int64_t),cudaMemcpyHostToDevice,m->stream))||!ok(cudaMemcpyAsync(m->dm,mask,rows,cudaMemcpyHostToDevice,m->stream)))return 2;
    if(phase_timings)cudaEventRecord(phases[1],m->stream);
    if(!timings&&m->graph_enabled){
        if(m->graph_exec&&m->graph_batch==batch&&m->graph_seq==seq){if(!ok(cudaGraphLaunch(m->graph_exec,m->stream)))return 2;m->graph_launches++;m->kernel_launches=1;m->last_graph_reused=true;}
        else if(m->graph_warmed&&m->graph_batch==batch&&m->graph_seq==seq){if(!ok(cudaStreamBeginCapture(m->stream,cudaStreamCaptureModeGlobal))||!launch_pipeline(m,batch,seq,rows,dn))return 2;cudaGraph_t captured{};if(!ok(cudaStreamEndCapture(m->stream,&captured)))return 2;cudaGraphExec_t executable{};if(!ok(cudaGraphInstantiate(&executable,captured,nullptr,nullptr,0))){cudaGraphDestroy(captured);return 2;}invalidate_graph(m);m->graph=captured;m->graph_exec=executable;m->graph_batch=batch;m->graph_seq=seq;m->graph_warmed=true;m->graph_captures++;if(!ok(cudaGraphLaunch(m->graph_exec,m->stream)))return 2;m->graph_launches++;m->kernel_launches=1;m->last_graph_captured=true;}
        else{invalidate_graph(m);m->graph_batch=batch;m->graph_seq=seq;m->graph_warmed=true;if(!launch_pipeline(m,batch,seq,rows,dn))return 2;}
    }else if(!timings){if(!launch_pipeline(m,batch,seq,rows,dn))return 2;}
    else{MARK_STAGE();embeddings<<<(dn+255)/256,256,0,m->stream>>>(m->di,m->dt,m->tmp,m->p[0],m->p[1],m->p[2],batch,seq);MARK_STAGE();layer_norm<<<rows,128,0,m->stream>>>(m->tmp,nullptr,m->a,m->p[3],m->p[4],rows);MARK_STAGE();for(int layer=0;layer<LAYERS;layer++){int x=5+layer*16;if(!linear(m,m->a,m->p[x],m->p[x+1],m->q,rows,D,D))return 2;MARK_STAGE();if(!linear(m,m->a,m->p[x+2],m->p[x+3],m->k,rows,D,D))return 2;MARK_STAGE();if(!linear(m,m->a,m->p[x+4],m->p[x+5],m->v,rows,D,D))return 2;MARK_STAGE();fused_attention<<<batch*H*seq,128,0,m->stream>>>(m->q,m->k,m->v,m->dm,m->merged,batch,seq);MARK_STAGE();MARK_STAGE();if(!linear(m,m->merged,m->p[x+6],m->p[x+7],m->tmp,rows,D,D))return 2;MARK_STAGE();layer_norm<<<rows,128,0,m->stream>>>(m->tmp,m->a,m->b,m->p[x+8],m->p[x+9],rows);MARK_STAGE();if(!linear_gelu(m,m->b,m->p[x+10],m->p[x+11],m->inter,rows,D,I))return 2;MARK_STAGE();MARK_STAGE();if(!linear(m,m->inter,m->p[x+12],m->p[x+13],m->tmp,rows,I,D))return 2;MARK_STAGE();layer_norm<<<rows,128,0,m->stream>>>(m->tmp,m->b,m->a,m->p[x+14],m->p[x+15],rows);MARK_STAGE();}cls_l2<<<batch,128,0,m->stream>>>(m->a,m->result,batch,seq);MARK_STAGE();}
    if(phase_timings)cudaEventRecord(phases[2],m->stream);if(!ok(cudaMemcpyAsync(host_out,m->result,(size_t)batch*D*sizeof(float),cudaMemcpyDeviceToHost,m->stream)))return 2;MARK_STAGE();if(phase_timings)cudaEventRecord(phases[3],m->stream);if(!ok(cudaStreamSynchronize(m->stream)))return 2;
    if(timings){*timing_count=marks-1;for(int i=0;i<marks-1;i++)cudaEventElapsedTime(&timings[i],events[i],events[i+1]);for(int i=0;i<marks;i++)cudaEventDestroy(events[i]);}
    if(phase_timings){for(int i=0;i<3;i++)cudaEventElapsedTime(&phase_timings[i],phases[i],phases[i+1]);for(int i=0;i<4;i++)cudaEventDestroy(phases[i]);}return 0;
}
extern "C" int cuda_bge_forward_impl(void* handle,const int64_t* ids,const uint8_t* mask,const int64_t* types,int batch,int seq,float* host_out){return cuda_bge_forward_core(handle,ids,mask,types,batch,seq,host_out,nullptr,0,nullptr,nullptr);}
extern "C" int cuda_bge_profile_impl(void* handle,const int64_t* ids,const uint8_t* mask,const int64_t* types,int batch,int seq,float* host_out,float* timings,int timing_capacity,int* timing_count){return cuda_bge_forward_core(handle,ids,mask,types,batch,seq,host_out,timings,timing_capacity,timing_count,nullptr);}
extern "C" int cuda_bge_forward_detailed_impl(void* handle,const int64_t* ids,const uint8_t* mask,const int64_t* types,int batch,int seq,float* host_out,float* phases){return phases?cuda_bge_forward_core(handle,ids,mask,types,batch,seq,host_out,nullptr,0,nullptr,phases):1;}
extern "C" void cuda_bge_destroy_impl(void* handle){Model*m=(Model*)handle;if(!m)return;release_workspace(m);for(float*&p:m->p)if(p)cudaFree(p);if(m->blas)cublasDestroy(m->blas);if(m->stream)cudaStreamDestroy(m->stream);delete m;}
extern "C" int cuda_bge_memory_info_impl(size_t* free_bytes,size_t* total_bytes){if(!free_bytes||!total_bytes)return 1;return ok(cudaMemGetInfo(free_bytes,total_bytes))?0:1;}
extern "C" int cuda_bge_diagnostics_impl(void* handle,uint64_t* values,size_t capacity){Model*m=(Model*)handle;if(!m||!values||capacity<18)return 1;uint64_t result[18]={m->forward_mallocs,m->forward_frees,m->kernel_launches,m->parameter_uploads,m->workspace_reallocations,m->h2d_bytes,m->d2h_bytes,m->synchronizations,m->graph_launches,m->internal_submissions,m->graph_captures,m->graph_invalidations,(uint64_t)m->last_batch,(uint64_t)m->last_seq,(uint64_t)m->graph_enabled,(uint64_t)m->last_graph_captured,(uint64_t)m->last_graph_reused,(uint64_t)(m->graph_exec!=nullptr)};for(int i=0;i<18;i++)values[i]=result[i];return 0;}
