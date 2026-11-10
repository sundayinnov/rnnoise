#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rnn.h"
#include "rnnoise_data.h"

// 原版数据文件里定义的数组名通常是 rnnoise_arrays
extern const WeightArray rnnoise_arrays[];
int init_rnnoise(RNNoise *model, const WeightArray *arrays);
void compute_rnn(const RNNoise *model, RNNState *rnn, float *gains, float *vad, const float *input, int arch);

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("用法: %s <input_f32> <output_raw>\n", argv[0]);
        return 1;
    }
    
    FILE *fin = fopen(argv[1], "rb");
    FILE *fout = fopen(argv[2], "wb");
    if (!fin || !fout) {
        printf("文件打开失败！\n");
        return 1;
    }

    RNNoise model;
    init_rnnoise(&model, rnnoise_arrays);
    
    RNNState rnn;
    memset(&rnn, 0, sizeof(rnn));

    float input[42];
    float gains[22];
    float vad;
    int frame_count = 0;
    
    printf("开始提取 384 维软标签...\n");
    while (fread(input, sizeof(float), 42, fin) == 42) {
        compute_rnn(&model, &rnn, gains, &vad, input, 0);
        fwrite(gains, sizeof(float), 22, fout);
        fwrite(&vad, sizeof(float), 1, fout);
        
        frame_count++;
        if (frame_count % 100000 == 0) {
            printf("已处理: %d 帧\n", frame_count);
        }
    }
    
    fclose(fin);
    fclose(fout);
    printf("✅ 完成！共处理 %d 帧，已保存至 %s\n", frame_count, argv[2]);
    return 0;
}
