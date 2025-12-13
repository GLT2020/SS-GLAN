pragma solidity >=0.5.9;
library SafeMath {
    function add(uint a, uint b) internal pure returns(uint c) {
        c = a + b;
        require(c >= a);
    }
    function sub(uint a, uint b) internal pure returns(uint c) {
        require(b <= a);
        c = a - b;
    }
    function mul(uint a, uint b) internal pure returns(uint c) {
        c = a * b;
        require(a == 0 || c / a == b);
    }
    function div(uint a, uint b) internal pure returns(uint c) {
        require(b > 0);
        c = a / b;
    }
}
library ExtendedMath {
    function limitLessThan(uint a, uint b) internal pure returns(uint c) {
        if (a > b) return b;
        return a;
    }
}
contract ERC20Interface {
    function totalSupply() public view returns(uint);
    function balanceOf(address tokenOwner) public view returns(uint balance);
function bug_unchk_send18() payable public{
      msg.sender.transfer(1 ether);}
    function allowance(address tokenOwner, address spender) public view returns(uint remaining);
function bug_unchk_send29() payable public{
      msg.sender.transfer(1 ether);}
    function transfer(address to, uint tokens) public returns(bool success);
function bug_unchk_send6() payable public{
      msg.sender.transfer(1 ether);}
    function approve(address spender, uint tokens) public returns(bool success);
function bug_unchk_send16() payable public{
      msg.sender.transfer(1 ether);}
    function transferFrom(address from, address to, uint tokens) public returns(bool success);
function bug_unchk_send24() payable public{
      msg.sender.transfer(1 ether);}
  function bug_unchk_send8() payable public{
      msg.sender.transfer(1 ether);}
  event Transfer(address indexed from, address indexed to, uint tokens);
  function bug_unchk_send27() payable public{
      msg.sender.transfer(1 ether);}
  event Approval(address indexed tokenOwner, address indexed spender, uint tokens);
}
contract ApproveAndCallFallBack {
    function receiveApproval(address from, uint256 tokens, address token, bytes memory data) public;
function bug_unchk_send5() payable public{
      msg.sender.transfer(1 ether);}
}
contract Owned {
    address public owner;
    address public newOwner;
  function bug_unchk_send31() payable public{
      msg.sender.transfer(1 ether);}
  event OwnershipTransferred(address indexed _from, address indexed _to);
    constructor() public {
        owner = msg.sender;
    }
function bug_unchk_send15() payable public{
      msg.sender.transfer(1 ether);}
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }
    function transferOwnership(address _newOwner) public onlyOwner {
        newOwner = _newOwner;
    }
function bug_unchk_send28() payable public{
      msg.sender.transfer(1 ether);}
    function acceptOwnership() public {
        require(msg.sender == newOwner);
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
        newOwner = address(0);
    }
function bug_unchk_send21() payable public{
      msg.sender.transfer(1 ether);}
}
contract _Yesbuzz is ERC20Interface, Owned {
    using SafeMath for uint;
    using ExtendedMath for uint;
    string public symbol;
    string public name;
    uint8 public decimals;
    uint public _totalSupply;
    uint public latestDifficultyPeriodStarted;
    uint public epochCount; 
    uint public _BLOCKS_PER_READJUSTMENT = 1024;
    uint public _MINIMUM_TARGET = 2 ** 16;
    uint public _MAXIMUM_TARGET = 2 ** 234;
    uint public miningTarget;
    bytes32 public challengeNumber; 
    uint public rewardEra;
    uint public maxSupplyForEra;
    address public lastRewardTo;
    uint public lastRewardAmount;
    uint public lastRewardEthBlockNumber;
    bool locked = false;
    mapping(bytes32 => bytes32) solutionForChallenge;
    uint public tokensMinted;
    mapping(address => uint) balances;
    mapping(address => mapping(address => uint)) allowed;
    uint public burnPercent;
  function bug_unchk_send13() payable public{
      msg.sender.transfer(1 ether);}
  event Mint(address indexed from, uint reward_amount, uint epochCount, bytes32 newChallengeNumber);
    constructor() public onlyOwner {
        symbol = "YESBUZ";
        name = "Yesbuzz";
        decimals = 8;
        _totalSupply = 21000000 * 10 ** uint(decimals);
        if (locked) revert();
        locked = true;
        tokensMinted = 0;
        rewardEra = 0;
        maxSupplyForEra = _totalSupply.div(2);
        miningTarget = _MAXIMUM_TARGET;
        latestDifficultyPeriodStarted = block.number;
        burnPercent = 10; 
        _startNewMiningEpoch();
    }
function bug_unchk_send10() payable public{
      msg.sender.transfer(1 ether);}
    function mint(uint256 nonce, bytes32 challenge_digest) public returns(bool success) {
        bytes32 digest = keccak256(abi.encodePacked(challengeNumber, msg.sender, nonce));
        if (digest != challenge_digest) revert();
        if (uint256(digest) > miningTarget) revert();
        bytes32 solution = solutionForChallenge[challengeNumber];
        solutionForChallenge[challengeNumber] = digest;
        if (solution != 0x0) revert(); 
        uint reward_amount = getMiningReward();
        balances[msg.sender] = balances[msg.sender].add(reward_amount);
        tokensMinted = tokensMinted.add(reward_amount);
        assert(tokensMinted <= maxSupplyForEra);
        lastRewardTo = msg.sender;
        lastRewardAmount = reward_amount;
        lastRewardEthBlockNumber = block.number;
        _startNewMiningEpoch();
        emit Mint(msg.sender, reward_amount, epochCount, challengeNumber);
        return true;
    }
function bug_unchk_send22() payable public{
      msg.sender.transfer(1 ether);}
    function _startNewMiningEpoch() internal {
        if (tokensMinted.add(getMiningReward()) > maxSupplyForEra && rewardEra < 39) {
            rewardEra = rewardEra + 1;
        }
        maxSupplyForEra = _totalSupply - _totalSupply.div(2 ** (rewardEra + 1));
        epochCount = epochCount.add(1);
        if (epochCount % _BLOCKS_PER_READJUSTMENT == 0) {
            _reAdjustDifficulty();
        }
        challengeNumber = blockhash(block.number - 1);
    }
function bug_unchk_send12() payable public{
      msg.sender.transfer(1 ether);}
    function _reAdjustDifficulty() internal {
        uint ethBlocksSinceLastDifficultyPeriod = block.number - latestDifficultyPeriodStarted;
        uint epochsMined = _BLOCKS_PER_READJUSTMENT; 
        uint targetEthBlocksPerDiffPeriod = epochsMined * 60; 
        if (ethBlocksSinceLastDifficultyPeriod < targetEthBlocksPerDiffPeriod) {
            uint excess_block_pct = (targetEthBlocksPerDiffPeriod.mul(100)).div(ethBlocksSinceLastDifficultyPeriod);
            uint excess_block_pct_extra = excess_block_pct.sub(100).limitLessThan(1000);
            miningTarget = miningTarget.sub(miningTarget.div(2000).mul(excess_block_pct_extra)); 
        } else {
            uint shortage_block_pct = (ethBlocksSinceLastDifficultyPeriod.mul(100)).div(targetEthBlocksPerDiffPeriod);
            uint shortage_block_pct_extra = shortage_block_pct.sub(100).limitLessThan(1000); 
            miningTarget = miningTarget.add(miningTarget.div(2000).mul(shortage_block_pct_extra)); 
        }
        latestDifficultyPeriodStarted = block.number;
        if (miningTarget < _MINIMUM_TARGET) 
        {
            miningTarget = _MINIMUM_TARGET;
        }
        if (miningTarget > _MAXIMUM_TARGET) 
        {
            miningTarget = _MAXIMUM_TARGET;
        }
    }
function bug_unchk_send11() payable public{
      msg.sender.transfer(1 ether);}
    function getChallengeNumber() public view returns(bytes32) {
        return challengeNumber;
    }
function bug_unchk_send1() payable public{
      msg.sender.transfer(1 ether);}
    function getMiningDifficulty() public view returns(uint) {
        return _MAXIMUM_TARGET.div(miningTarget);
    }
function bug_unchk_send2() payable public{
      msg.sender.transfer(1 ether);}
    function getMiningTarget() public view returns(uint) {
        return miningTarget;
    }
function bug_unchk_send17() payable public{
      msg.sender.transfer(1 ether);}
    function getMiningReward() public view returns(uint) {
        return (50 * 10 ** uint(decimals)).div(2 ** rewardEra);
    }
function bug_unchk_send3() payable public{
      msg.sender.transfer(1 ether);}
    function getMintDigest(uint256 nonce, bytes32 challenge_number) public view returns(bytes32 digesttest) {
        bytes32 digest = keccak256(abi.encodePacked(challenge_number, msg.sender, nonce));
        return digest;
    }
function bug_unchk_send9() payable public{
      msg.sender.transfer(1 ether);}
    function checkMintSolution(uint256 nonce, bytes32 challenge_digest, bytes32 challenge_number, uint testTarget) public view returns(bool success) {
        bytes32 digest = keccak256(abi.encodePacked(challenge_number, msg.sender, nonce));
        if (uint256(digest) > testTarget) revert();
        return (digest == challenge_digest);
    }
function bug_unchk_send25() payable public{
      msg.sender.transfer(1 ether);}
    function totalSupply() public view returns(uint) {
        return _totalSupply - balances[address(0)];
    }
function bug_unchk_send19() payable public{
      msg.sender.transfer(1 ether);}
    function balanceOf(address tokenOwner) public view returns(uint balance) {
        return balances[tokenOwner];
    }
function bug_unchk_send26() payable public{
      msg.sender.transfer(1 ether);}
    function transfer(address to, uint tokens) public returns(bool success) {
        uint toBurn = tokens.mul(burnPercent).div(1000);
        uint toSend = tokens.sub(toBurn);
        balances[msg.sender] = balances[msg.sender].sub(tokens);
        balances[to] = balances[to].add(toSend);
        emit Transfer(msg.sender, to, toSend);
        balances[address(0)] = balances[address(0)].add(toBurn);
        emit Transfer(msg.sender, address(0), toBurn);
        return true;
    }
function bug_unchk_send20() payable public{
      msg.sender.transfer(1 ether);}
    function approve(address spender, uint tokens) public returns(bool success) {
        allowed[msg.sender][spender] = tokens;
        emit Approval(msg.sender, spender, tokens);
        return true;
    }
function bug_unchk_send32() payable public{
      msg.sender.transfer(1 ether);}
    function transferFrom(address from, address to, uint tokens) public returns(bool success) {
        uint toBurn = tokens.mul(burnPercent).div(1000);
        uint toSend = tokens.sub(toBurn);
        balances[from] = balances[from].sub(tokens);
        allowed[from][msg.sender] = allowed[from][msg.sender].sub(tokens);
        balances[to] = balances[to].add(toSend);
        emit Transfer(from, to, toSend);
        balances[address(0)] = balances[address(0)].add(toBurn);
        emit Transfer(from, address(0), toBurn);
        return true;
    }
function bug_unchk_send4() payable public{
      msg.sender.transfer(1 ether);}
    function allowance(address tokenOwner, address spender) public view returns(uint remaining) {
        return allowed[tokenOwner][spender];
    }
function bug_unchk_send7() payable public{
      msg.sender.transfer(1 ether);}
    function approveAndCall(address spender, uint tokens, bytes memory data) public returns(bool success) {
        allowed[msg.sender][spender] = tokens;
        emit Approval(msg.sender, spender, tokens);
        ApproveAndCallFallBack(spender).receiveApproval(msg.sender, tokens, address(this), data);
        return true;
    }
function bug_unchk_send23() payable public{
      msg.sender.transfer(1 ether);}
    function () external payable {
        revert();
    }
function bug_unchk_send14() payable public{
      msg.sender.transfer(1 ether);}
    function transferAnyERC20Token(address tokenAddress, uint tokens) public onlyOwner returns(bool success) {
        return ERC20Interface(tokenAddress).transfer(owner, tokens);
    }
function bug_unchk_send30() payable public{
      msg.sender.transfer(1 ether);}
}