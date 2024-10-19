## 哈希

[1. 两数之和 - 力扣（LeetCode）](https://leetcode.cn/problems/two-sum/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer,Integer> val2Index = new HashMap<>();
        // 遍历每个元素
        for(int i=0;i<nums.length;i++){
            int x = nums[i];
            // 如果之前遍历的元素存在值为target-x的，直接返回结果
            if(val2Index.containsKey(target - x)){
                return new int[]{i,val2Index.get(target - x)};
            }
            // 记录当前元素的值和下标
            val2Index.put(x,i);
        }
        return null;
    }
}
```

[49. 字母异位词分组 - 力扣（LeetCode）](https://leetcode.cn/problems/group-anagrams/?envType=study-plan-v2&envId=top-100-liked)

思路：题目要将字符串数组中的字母异位词（由重新排列源单词所有字母得到的一个新单词）组合在一起，并返回结果。如何判断两个单词是异位词呢，可以对单词进行编码，一共有26个字母，因此可以使用code[]数组统计每个字母的出现次数，将code数组转为字符串即为编码值。最后使用HashMap存储结果，键为对应异位词的编码值，值为编码值为对应编码值的所有字符串列表。

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        // 编码-分组映射
        Map<String,List<String>> code2Group=new HashMap<>();
        // 遍历每一种编码，将相同编码的元素添加到同一组中
        for(String str:strs){
            String code=encode(str);
            code2Group.putIfAbsent(code,new ArrayList<>());
            code2Group.get(code).add(str);
        }
        List<List<String>> res=new ArrayList<>();
        for(List<String> group:code2Group.values()){
            res.add(group);
        }
        return res;
    }
    //利用每个字符的出现次数进行编码
    String encode(String str){
        char[] code=new char[26];
        for(char c:str.toCharArray()){
            code[c-'a']++;
        }
        return new String(code);
    }
}
```

[128. 最长连续序列 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-consecutive-sequence/description/?envType=study-plan-v2&envId=top-100-liked)

思路一：

- 将所有元素添加到集合set中，便于判断该元素是否存在。
- 遍历所有的元素，对于每个元素，循环查找以当前元素开始的连续元素，并记录连续序列的长度
- 设当前元素值为x，如果存在值为x-1的元素，直接跳过（不是连续序列的开始位置）

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        //将所有元素添加到集合，便于查看是否存在
        Set<Integer> set=new HashSet<>();
        for(int x:nums){
            set.add(x);
        }
        int res=0;
        //寻找每一个序列
        for(int x:set){
            // 存在比当前值x更小的，不是序列开始位置
            if(set.contains(x-1)) continue;
            int len=1;
            // 循环查找比当前元素大1的元素
            while(set.contains(x+1)){
                len++;
                x++;
            }
            res=Math.max(res,len);
        }
        return res;
    }
    
}
```

思路二：

将数组去重并排序，遍历数组，每次以当前位置i作为连续序列的起始位置，并记录连续序列的长度，再将指针移到该连续序列的末尾，开始查找下一个连续序列。

```java
class Solution {
	public int longestConsecutive(int[] nums) {
        //将数组去重
        Set<Integer> s = new HashSet<>();
        for(int x:nums){
            s.add(x);
        }
        Integer[] nums2 = s.toArray(new Integer[0]);
        //将数组排序
        Arrays.sort(nums2);
        int n = nums2.length;
        int i=0;
        int len = 0;
		//遍历每个元素
        while(i<n){
            int j=i;
            //查找以nums[i]为起点的连续序列
            while(j+1<n && nums2[j+1]-nums2[j]==1) j++;
            len = Math.max(len,j-i+1);
            //移到下一个连续序列的起始位置
            i = j+1;
        }
        return len;
    }
}
```

## 双指针

[283. 移动零 - 力扣（LeetCode）](https://leetcode.cn/problems/move-zeroes/description/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public void moveZeroes(int[] nums) {
        int i = 0;
        // 将非0数字移到前面
        for(int j=0;j<nums.length;j++){
            if(nums[j] != 0){
                nums[i++] = nums[j];
            }
        }
        // 将后面的元素全部置0
        while(i<nums.length) nums[i++] = 0;
    }
}
```

[11. 盛最多水的容器 - 力扣（LeetCode）](https://leetcode.cn/problems/container-with-most-water/description/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public int maxArea(int[] height) {
        int left=0,right=height.length-1;
        int res=0;
        while(left<right){
            int t=Math.min(height[left],height[right])*(right-left);
            res=Math.max(res,t);
            if(height[left]<height[right]) left++;
            else right--;
        }
        return res;
    }
}
```

[15. 三数之和 - 力扣（LeetCode）](https://leetcode.cn/problems/3sum/description/?envType=study-plan-v2&envId=top-100-liked)

思路一：首先将数组排序，枚举三元组的前2个元素的位置i, j，并在区间 [j+1,n-1] 二分查找第三个元素。由于结果不能重复，需要使用集合保存结果。

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Set<List<Integer>> res = new HashSet<>();
        //将数组排序
        Arrays.sort(nums);
        int n = nums.length;
        //枚举三元组前2个元素
        for(int i=0;i<n-2;i++){
            for(int j=i+1;j<n-1;j++){
                //二分查找第3个元素
                int k = bs(nums,j+1,n-1,-(nums[i]+nums[j]));
                if(k!=-1) res.add(new ArrayList<>(Arrays.asList(new Integer[]{nums[i],nums[j],nums[k]})));
            }
        }
        return new ArrayList<List<Integer>>(res);
    }

    int bs(int[] nums,int l,int r,int target){
        while(l<=r){
            int mid = (l+r)>>1;
            if(nums[mid] == target) return mid;
            else if(nums[mid] > target) r = mid-1;
            else l = mid+1;
        }
        return -1;
    }
}
```

思路二：

直接转换为nSum问题，遍历n元组开始元素的位置i, 递归求目标值为 target-nums[i] 的 (n-1) 元组，当n=2时，直接使用双指针求出结果即可。

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        // 将数组排序
        Arrays.sort(nums);
        return nSum(nums,3,0,0);
    }

    private List<List<Integer>> nSum(int[] nums,int n,int start,int target){
        int sz = nums.length;
        List<List<Integer>> res = new ArrayList<>();
        // 至少是 2Sum，且搜索区间大小sz-start不小于n
        if(n<2 || sz-start<n) return res;
        // base case
        if(n==2){
            int lo = start,hi = sz-1;
            while(lo < hi){
                int left = nums[lo],right = nums[hi];
                int sum = left + right;
                if(sum < target){ //和较小，右移左指针
                    while(lo < hi && nums[lo] == left) lo++;
                }else if(sum > target){ //和较大，左移右指针
                    while(lo < hi && nums[hi] == right) hi--;
                }else{
                    //记录二元组
                    res.add(new ArrayList(Arrays.asList(new Integer[]{left,right})));
                    //更新左右指针的位置
                    while(lo < hi && nums[lo] == left) lo++;
                    while(lo < hi && nums[hi] == right) hi--;
                }
            }
        }else{
            // 遍历元组的开始位置
            for(int i=start;i<sz;i++){
                // 递归计算子结果
                List<List<Integer>> sub = nSum(nums,n-1,i+1,target-nums[i]);
                // 将开始元素添加到子结果中
                for(List<Integer> list:sub){
                    list.add(nums[i]);
                    res.add(list);
                }
                // 跳过重复元素
                while(i < sz-1 && nums[i] == nums[i+1]) i++;
            }
        }
        
        return res;
    }
}
```

[42. 接雨水 - 力扣（LeetCode）](https://leetcode.cn/problems/trapping-rain-water/description/?envType=study-plan-v2&envId=top-100-liked)

思路一：

每个柱子可以接到的雨水为当前柱子左边柱子的最大高度和右边柱子的最大高度的较小值减去当前柱子的高度，即：$water[i] = min(max(height[0..i],height[i..n-1]))-height[i]$，因此，直接计算出每个柱子左右两边的最大柱子高度lmax[i], rmax[i]，最后计算每个柱子可以接的雨水即可。

```java
class Solution {
    public int trap(int[] height) {
        int n=height.length;
        // lmax[i]: [0..i]区间的柱子高度的最大值
        int[] lmax = new int[n];
        // rmax[i]: [i..n-1]区间的柱子高度的最大值
        int[] rmax = new int[n];
        lmax[0] = height[0]; 
        rmax[n-1] = height[n-1];
        for(int i=1;i<n;i++){
            lmax[i] = Math.max(lmax[i-1],height[i]);
        }
        for(int i=n-2;i>=0;i--){
            rmax[i] = Math.max(rmax[i+1],height[i]);
        }
        int res = 0;
        // 计算每个柱子上可以接的雨水
        for(int i=0;i<n;i++){
            res += Math.min(lmax[i], rmax[i]) - height[i];
        }
        return res;
    }
}
```

思路二：

根据每个柱子接雨水的高度公式可以看出，当前柱子能接的水量是由左右最高柱子高度的较小值决定的，因此可以使用双指针left, right，在计算lmax[0..left]和rmax[right..n-1]的同时，计算每个柱子可以接的雨水：

- 若left < right：此时rmax[right..n-1] < rmax[left..n-1]，能接的雨水由lmax决定，res+=lmax-heigth[left]
- 若left > right：此时lmax[0..left] < lmax[0..right]，能接的雨水由rmax决定，res+=rmax-heigth[right]

```java
class Solution {
    public int trap(int[] height) {
        int n=height.length;
        int lmax=0,rmax=0;
        int left=0,right=n-1;
        int res=0;
        // 注意此处可以为<，因为最终height[left]=height[right]=maxHeight
        while(left<=right){
            lmax=Math.max(lmax,height[left]);
            rmax=Math.max(rmax,height[right]);
            if(lmax<rmax){
                res+=lmax-height[left];
                left++;
            }else{
                res+=rmax-height[right];
                right--;
            }
        }
        return res;
    }
}
```

